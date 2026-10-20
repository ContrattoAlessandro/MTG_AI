package org.mage.magezero;

import mage.cards.decks.*;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.*;
import mage.constants.*;
import mage.game.*;
import mage.game.match.*;
import mage.game.mulligan.Mulligan;
import mage.game.mulligan.MulliganType;
import mage.player.ai.policy.PolicyPlayer;
import mage.player.ai.policy.RecordingPlayer;
import mage.player.ai.policy.WebPlayer;
import mage.players.Player;
import mage.util.RandomUtil;
import mage.util.ThreadUtils;
import com.google.gson.Gson;
import mage.player.ai.TeacherPlayer;
import mage.player.ai.XMageAiPlayer;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Plays batches of policy games in XMage.
 *
 * {@code PolicyGames game.json} plays one batch and exits. {@code PolicyGames --serve}
 * boots once and then plays one batch per game file path read from stdin, answering
 * each with a single {@code MZ-POLICY RESULT terminal requested truncated errors timeouts}
 * or {@code MZ-POLICY ERROR} line, so the card database, card scan and warmed JIT
 * survive across generations.
 *
 * A failing game never fails its batch: rules-engine exceptions and wall-clock
 * timeouts are counted and their trajectories discarded. Games stopped by the turn
 * limit are reported as truncated; their trajectories are kept for bootstrapping.
 */
public final class PolicyGames {
    static final String TAG = "MZ-POLICY";
    static final String XMAGE_AI = "xmage-ai";
    // XMage's AI playing while its decisions are recorded for behaviour cloning.
    static final String TEACHER = "xmage-teacher";
    static final String COMMANDER = "commander";
    // A person in a browser (WebPlayer); server.web_endpoint names the `mz web` server.
    static final String WEB = "web";
    private static final ScheduledExecutorService WATCHDOG = Executors.newSingleThreadScheduledExecutor(runnable -> {
        Thread thread = new Thread(runnable, "policy-game-watchdog");
        thread.setDaemon(true);
        return thread;
    });

    enum Outcome { TERMINAL, TRUNCATED, ERROR, TIMEOUT }

    public static void main(String[] args) throws Exception {
        // XMage's AI players share one pool of search threads (5): with more concurrent
        // gauntlet games they queue on it and time out.
        XMageAiPlayer.configureSimulationThreads(Integer.getInteger("xmage.ai.simulationThreads",
            Runtime.getRuntime().availableProcessors()));
        if (args.length == 1 && "--serve".equals(args[0])) {
            serve();
            return;
        }
        if (args.length != 1) throw new IllegalArgumentException("usage: PolicyGames game.json | PolicyGames --serve");
        bootstrap();
        int[] result = runBatch(args[0]);
        System.out.println("Games terminal/requested/truncated/errors/timeouts: " + Arrays.toString(result));
    }

    private static void bootstrap() {
        RepositoryUtil.bootstrapLocalDb();
        CardScanner.scan();
    }

    private static void serve() throws IOException {
        bootstrap();
        BufferedReader commands = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        reply("READY");
        String line;
        while ((line = commands.readLine()) != null) {  // EOF from the runner ends the engine
            if (line.trim().isEmpty()) continue;
            try {
                int[] result = runBatch(line.trim());
                reply("RESULT " + result[0] + " " + result[1] + " " + result[2] + " " + result[3] + " " + result[4]);
            } catch (Throwable error) {
                error.printStackTrace();
                reply("ERROR " + String.valueOf(error).replace('\r', ' ').replace('\n', ' '));
            }
        }
    }

    private static void reply(String message) {
        System.out.println(TAG + " " + message);
        System.out.flush();
    }

    /** Plays one configured batch; returns {terminal, requested, truncated, errors, timeouts}. */
    private static int[] runBatch(String configPath) throws Exception {
        Map<String, Object> raw = rawConfig(configPath);
        GameConfig config = new GameConfig(raw);
        String typeA = config.typeA, typeB = config.typeB;
        if (!("policy".equals(typeA) || TEACHER.equals(typeA))
                || !("policy".equals(typeB) || XMAGE_AI.equals(typeB) || TEACHER.equals(typeB) || WEB.equals(typeB))) {
            throw new IllegalArgumentException("player_a must be type policy or " + TEACHER
                + ", player_b policy, " + XMAGE_AI + ", " + TEACHER + " or " + WEB);
        }
        if (!"normal".equals(config.gameMode) && !COMMANDER.equals(config.gameMode)) {
            throw new IllegalArgumentException("policy games support game_mode normal (two-player constructed) or "
                + COMMANDER + " (Commander Duel)");
        }
        if (config.seesOpponentHand) {
            throw new IllegalArgumentException("policy games require hiddenInfo.see_opponent_hand: false for both players");
        }
        if (config.manualTap) {
            throw new IllegalArgumentException("initial policy adapter requires automatic engine mana payment");
        }
        if (config.threads < 1 || config.threads > 16
                || config.games < 1 || config.maxMinutes < 1
                || config.maxTurns < 1) throw new IllegalArgumentException("invalid collection limits (threads 1..16)");
        // Optional policy-service shards: game i uses shard i mod n for both players.
        List<Integer> ports = shardPorts(raw, "ports", config.port);
        List<Integer> opponentPorts = shardPorts(raw, "opponent_ports", config.opponentPort);
        Map<String, Object> server = section(raw, "server");
        // Odd games swap the two controllers between seats, so a learner facing a
        // fixed opponent plays both decks of an asymmetric matchup. With
        // learner_keeps_deck the deck moves with its controller: the learner always
        // plays player_a's deck, and only the seat (and starting order) changes.
        boolean alternate = Boolean.TRUE.equals(server.get("alternate_seats"));
        boolean keepDeck = Boolean.TRUE.equals(server.get("learner_keeps_deck"));
        // sample: false makes policy players take their most likely action (greedy
        // evaluation); the policy service never records unsampled games.
        boolean sample = !Boolean.FALSE.equals(server.get("sample"));
        int aiSkill = ((Number) section(raw, "opponent_ai").getOrDefault("skill", 6)).intValue();
        if (aiSkill < 1 || aiSkill > 10) throw new IllegalArgumentException("opponent_ai.skill must be in 1..10");
        // Optional cap on the AI's search time per decision (XMage default: 3 s per skill level).
        int aiThink = ((Number) section(raw, "opponent_ai").getOrDefault("think_seconds", 0)).intValue();
        if (aiThink < 0) throw new IllegalArgumentException("opponent_ai.think_seconds must be >= 0 (0 = XMage default)");
        // A single turn running longer than this is a loop (XMage's AI repeating an
        // untap/mana activation forever); the game is stopped as a timeout.
        int turnSeconds = ((Number) section(raw, "training").getOrDefault("max_turn_seconds", 120)).intValue();
        String webEndpoint = (String) server.get("web_endpoint");
        if (WEB.equals(typeB) && (webEndpoint == null || config.games != 1 || alternate)) {
            throw new IllegalArgumentException("a web game needs server.web_endpoint, one game and no seat alternation");
        }
        if (turnSeconds < 0) throw new IllegalArgumentException("training.max_turn_seconds must be >= 0 (0 = no limit)");
        // Import lists once; instantiate independent card/deck objects in each game.
        DeckCardLists a = DeckImporter.importDeckFromFile(config.deckA, true);
        DeckCardLists b = DeckImporter.importDeckFromFile(config.deckB, true);
        // XMage runs game code only on threads it names as game threads (ThreadUtils).
        java.util.concurrent.atomic.AtomicInteger threadNumber = new java.util.concurrent.atomic.AtomicInteger();
        ExecutorService workers = Executors.newFixedThreadPool(config.threads,
            runnable -> new Thread(runnable, ThreadUtils.THREAD_PREFIX_GAME + " policy-" + threadNumber.incrementAndGet()));
        List<Future<Outcome>> games = new ArrayList<>();
        for (int i = 0; i < config.games; i++) {
            final int gameNumber = i;
            final int port = ports.get(i % ports.size());
            final int opponentPort = opponentPorts.get(i % opponentPorts.size());
            final boolean swap = alternate && i % 2 == 1;
            games.add(workers.submit(() -> play(config, a, b, gameNumber, port, opponentPort, typeA, typeB, swap, keepDeck, sample, aiSkill, aiThink, turnSeconds, webEndpoint)));
        }
        workers.shutdown();
        int[] counts = new int[Outcome.values().length];
        // Each game interrupts itself after max_minutes. This deadline is only a
        // backstop for a game stuck outside policy calls (an engine loop).
        long rounds = (games.size() + config.threads - 1) / config.threads;
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(rounds * (config.maxMinutes + 1L));
        try {
            for (Future<Outcome> game : games) {
                Outcome outcome;
                try {
                    outcome = game.get(Math.max(0, deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
                } catch (TimeoutException error) {
                    game.cancel(true);
                    outcome = Outcome.TIMEOUT;
                } catch (ExecutionException error) {
                    error.getCause().printStackTrace();
                    outcome = Outcome.ERROR;
                }
                counts[outcome.ordinal()]++;
            }
        } finally {
            workers.shutdownNow();
        }
        return new int[] {counts[Outcome.TERMINAL.ordinal()], games.size(), counts[Outcome.TRUNCATED.ordinal()],
                          counts[Outcome.ERROR.ordinal()], counts[Outcome.TIMEOUT.ordinal()]};
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> rawConfig(String configPath) throws IOException {
        try (Reader reader = Files.newBufferedReader(Paths.get(configPath), StandardCharsets.UTF_8)) {
            Map<String, Object> raw = new Gson().fromJson(reader, Map.class);
            return raw == null ? Collections.emptyMap() : raw;
        }
    }

    /** The settings of one batch (a JSON game file written by the Python runner). */
    static final class GameConfig {
        final String goesFirst, gameMode, typeA, typeB, deckA, deckB, host;
        final boolean seesOpponentHand, manualTap;
        final int games, threads, maxTurns, maxMinutes, port, opponentPort;

        GameConfig(Map<String, Object> raw) {
            Map<String, Object> a = section(raw, "player_a"), b = section(raw, "player_b");
            Map<String, Object> training = section(raw, "training"), server = section(raw, "server");
            goesFirst = text(raw, "goes_first", "random");
            gameMode = text(raw, "game_mode", "normal");
            typeA = text(a, "type", "");
            typeB = text(b, "type", "");
            deckA = text(a, "deckPath", null);
            deckB = text(b, "deckPath", null);
            seesOpponentHand = flag(section(a, "hiddenInfo"), "see_opponent_hand", true)
                || flag(section(b, "hiddenInfo"), "see_opponent_hand", true);
            manualTap = flag(section(a, "gameplay"), "manual_tapping", false)
                || flag(section(b, "gameplay"), "manual_tapping", false);
            games = number(training, "games", 1000);
            threads = number(training, "threads", 2);
            maxTurns = number(training, "max_turns", 50);
            maxMinutes = number(training, "max_minutes", 20);
            host = text(server, "host", "localhost");
            port = number(server, "port", 8080);
            opponentPort = number(server, "opponent_port", 8081);
            if (deckA == null || deckB == null) throw new IllegalArgumentException("player_a and player_b need a deckPath");
        }

        private static String text(Map<String, Object> raw, String key, String fallback) {
            Object value = raw.get(key);
            return value == null ? fallback : String.valueOf(value);
        }

        private static boolean flag(Map<String, Object> raw, String key, boolean fallback) {
            Object value = raw.get(key);
            return value instanceof Boolean ? (Boolean) value : fallback;
        }

        private static int number(Map<String, Object> raw, String key, int fallback) {
            Object value = raw.get(key);
            return value instanceof Number ? ((Number) value).intValue() : fallback;
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> section(Map<String, Object> raw, String key) {
        Object value = raw.get(key);
        return value instanceof Map ? (Map<String, Object>) value : Collections.emptyMap();
    }

    private static List<Integer> shardPorts(Map<String, Object> raw, String key, int fallback) {
        Object listed = section(raw, "server").get(key);
        if (listed == null) return Collections.singletonList(fallback);
        if (!(listed instanceof List) || ((List<?>) listed).isEmpty()) {
            throw new IllegalArgumentException("server." + key + " must be a nonempty list of ports");
        }
        List<Integer> ports = new ArrayList<>();
        for (Object port : (List<?>) listed) {
            if (!(port instanceof Number)) throw new IllegalArgumentException("server." + key + " must contain port numbers");
            ports.add(((Number) port).intValue());
        }
        return ports;
    }

    /** XMage's own "Mad" AI (with the fixes in XMageAiPlayer). */
    private static Player xmageAi(String name, int skill, int thinkSeconds) {
        XMageAiPlayer player = new XMageAiPlayer(name, RangeOfInfluence.ONE, skill);
        if (thinkSeconds > 0) player.setMaxThinkTimeSecs(thinkSeconds);
        return player;
    }

    /** The AI above, recording its decisions through the policy service at `endpoint`. */
    private static Player teacher(String name, int skill, int thinkSeconds, String endpoint) {
        TeacherPlayer player = new TeacherPlayer(name, RangeOfInfluence.ONE, skill, endpoint);
        if (thinkSeconds > 0) player.setMaxThinkTimeSecs(thinkSeconds);
        return player;
    }

    /** Two-player game for the configured mode; the Commander classes load by name, like xmageAi. */
    private static Game newGame(GameConfig config) throws ReflectiveOperationException {
        Mulligan mulligan = MulliganType.GAME_DEFAULT.getMulligan(0);
        if (!COMMANDER.equals(config.gameMode)) {
            return new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE, mulligan, 60, 20, 7);
        }
        // Commander Duel as XMage's "Commander Two Player Duel" table creates it:
        // 100-card decks, 40 life, seven cards, 21 commander damage loses.
        GameCommanderImpl game = (GameCommanderImpl) Class.forName("mage.game.CommanderDuel")
            .getConstructor(MultiplayerAttackOption.class, RangeOfInfluence.class, Mulligan.class, int.class, int.class)
            .newInstance(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE, mulligan, 40, 7);
        game.setCheckCommanderDamage(true);
        return game;
    }

    private static Match newMatch(GameConfig config) throws ReflectiveOperationException {
        MatchOptions options = new MatchOptions("policy selfplay", "policy", false);
        if (!COMMANDER.equals(config.gameMode)) return new TwoPlayerMatch(options);
        return (Match) Class.forName("mage.game.CommanderDuelMatch").getConstructor(MatchOptions.class).newInstance(options);
    }

    private static Outcome play(GameConfig config, DeckCardLists listA, DeckCardLists listB, int index, int port, int opponentPort,
                                String learnerType, String opponentType, boolean swap, boolean keepDeck, boolean sample, int aiSkill,
                                int aiThink, int turnSeconds, String webEndpoint) throws Exception {
        RandomUtil.setSeed(ThreadLocalRandom.current().nextLong());
        Thread self = Thread.currentThread();
        Game game = newGame(config);
        if (Boolean.getBoolean("magezero.gameLog")) {
            // -Dmagezero.gameLog=true: the game's own log (as XMage clients show it) in the engine log.
            game.addTableEventListener(event -> {
                if (event.getMessage() != null && !event.getMessage().isEmpty()) {
                    System.err.println("[game " + index + "] T" + game.getTurnNum() + " " + event.getMessage().replaceAll("<[^>]+>", ""));
                }
            });
        }
        // Checked every second: total time over max_minutes, or one turn over
        // max_turn_seconds, stops the game as a timeout. The interrupt is repeated
        // because XMage's AI clears it while it waits for its own search.
        Object lock = new Object();
        boolean[] finished = {false};
        boolean[] timedOut = {false};
        long started = System.nanoTime();
        long[] turnStarted = {started};
        int[] turn = {-1};
        ScheduledFuture<?> watchdog = WATCHDOG.scheduleAtFixedRate(() -> {
            synchronized (lock) {
                if (finished[0]) return;
                long now = System.nanoTime();
                int current = game.getTurnNum();
                if (current != turn[0]) { turn[0] = current; turnStarted[0] = now; }
                if (timedOut[0] || now - started > TimeUnit.MINUTES.toNanos(config.maxMinutes)
                        || (turnSeconds > 0 && now - turnStarted[0] > TimeUnit.SECONDS.toNanos(turnSeconds))) {
                    timedOut[0] = true;
                    self.interrupt();
                }
            }
        }, 1, 1, TimeUnit.SECONDS);
        Match match = newMatch(config);
        String host = "http://" + config.host + ":";
        // The learner controller always uses `port`; swapping moves it to seat B.
        boolean web = WEB.equals(opponentType);
        String learnerName = web ? "MageZero" : swap ? "PlayerB" : "PlayerA";
        Player learner = TEACHER.equals(learnerType) ? teacher(learnerName, aiSkill, aiThink, host + port)
            : new PolicyPlayer(learnerName, RangeOfInfluence.ONE, host + port, sample);
        String opponentName = web ? "Human" : swap ? "PlayerA" : "PlayerB";
        Player opponent = web ? new WebPlayer(opponentName, RangeOfInfluence.ONE, host + port, webEndpoint)
            : XMAGE_AI.equals(opponentType) ? xmageAi(opponentName, aiSkill, aiThink)
            : TEACHER.equals(opponentType) ? teacher(opponentName, aiSkill, aiThink, host + opponentPort)
            : new PolicyPlayer(opponentName, RangeOfInfluence.ONE, host + opponentPort, sample);
        Player a = swap ? opponent : learner, b = swap ? learner : opponent;
        Outcome outcome = Outcome.ERROR;
        try {
            // Without keepDeck the decks stay with their seats; with it, with their controllers.
            boolean exchange = swap && keepDeck;
            load(game, match, a, Deck.load(exchange ? listB : listA, false, false, new HashMap<>()));
            load(game, match, b, Deck.load(exchange ? listA : listB, false, false, new HashMap<>()));
            if (a instanceof RecordingPlayer) ((RecordingPlayer) a).initializeKnownDeck(game, b.getId());
            if (b instanceof RecordingPlayer) ((RecordingPlayer) b).initializeKnownDeck(game, a.getId());
            GameOptions options = new GameOptions();
            // A real game: XMage's test mode is for its unit tests and deals no opening hands.
            // (Each player's own test mode, set in load(), only changes how errors are handled.)
            options.testMode = false;
            options.stopOnTurn = config.maxTurns;
            options.stopAtStep = PhaseStep.END_TURN;
            game.setGameOptions(options);
            if ("player_a".equals(config.goesFirst)) game.setStartingPlayerId(a.getId());
            else if ("player_b".equals(config.goesFirst)) game.setStartingPlayerId(b.getId());
            // Decorrelated from seat swapping: each controller starts with each deck.
            else game.setStartingPlayerId((index / 2) % 2 == 0 ? a.getId() : b.getId());
            game.start(null);
            // XMage swallows rules errors and ends test-mode games without a
            // winner, so only an error-free game that reached the turn limit is
            // a truncation; anything else unfinished is an error.
            if (self.isInterrupted() || timedOut[0]) outcome = Outcome.TIMEOUT;
            else if (game.getTotalErrorsCount() > 0) outcome = Outcome.ERROR;
            else if (a.hasWon() || b.hasWon() || (a.hasLost() && b.hasLost())) outcome = Outcome.TERMINAL;
            else if (game.getTurnNum() >= config.maxTurns) outcome = Outcome.TRUNCATED;
        } catch (Throwable error) {
            outcome = self.isInterrupted() || timedOut[0] ? Outcome.TIMEOUT : Outcome.ERROR;
            if (outcome == Outcome.ERROR) {
                System.err.println("policy game " + index + " failed:");
                error.printStackTrace();
            }
        }
        // Every Throwable is caught above, so this cleanup always runs.
        synchronized (lock) {
            finished[0] = true;
            watchdog.cancel(false);
        }
        System.err.println("policy game " + index + " " + outcome + ": turn " + game.getTurnNum()
            + "; " + summary(game, a, b) + "; " + summary(game, b, a));
        Thread.interrupted();  // never leak an interrupt into the next pooled game
        if (outcome == Outcome.ERROR && game.getTotalErrorsCount() > 0) {
            System.err.println("policy game " + index + " ended after " + game.getTotalErrorsCount() + " rules-engine error(s)");
        }
        for (Player player : Arrays.asList(a, b)) {
            if (!(player instanceof RecordingPlayer)) continue;
            try {
                ((RecordingPlayer) player).finish(game, outcome == Outcome.TERMINAL, outcome == Outcome.TRUNCATED);
            } catch (RuntimeException error) {
                // A failed /finish discards this trajectory; it must not fail the batch.
                System.err.println("policy game " + index + " finish failed: " + error);
                if (outcome == Outcome.TERMINAL || outcome == Outcome.TRUNCATED) outcome = Outcome.ERROR;
            }
        }
        return outcome;
    }

    /** End state of one player, for the engine log: why games end (life, poison, commander damage). */
    private static String summary(Game game, Player player, Player other) {
        int commanderDamage = 0;
        for (UUID commander : game.getCommandersIds(other, CommanderCardType.ANY, false)) {
            mage.watchers.common.CommanderInfoWatcher watcher =
                game.getState().getWatcher(mage.watchers.common.CommanderInfoWatcher.class, commander);
            if (watcher != null) commanderDamage = Math.max(commanderDamage, watcher.getDamageToPlayer().getOrDefault(player.getId(), 0));
        }
        return player.getName() + (player instanceof WebPlayer ? " (web)" : player instanceof PolicyPlayer ? " (policy)" : player instanceof RecordingPlayer ? " (teacher)" : "") + " "
            + (player.hasWon() ? "won" : player.hasLost() ? "lost" : "-") + " life " + player.getLife()
            + " poison " + player.getCountersCount(mage.counters.CounterType.POISON)
            + " commander damage " + commanderDamage + " library " + player.getLibrary().size();
    }

    private static void load(Game game, Match match, Player player, Deck deck) {
        player.setTestMode(true);
        player.setFastFailInTestMode(false); // real-game behaviour: skip impossible AI blocks instead of throwing
        game.loadCards(deck.getCards(), player.getId());
        game.loadCards(deck.getSideboard(), player.getId());
        game.addPlayer(player, deck);
        match.addPlayer(player, deck);
    }
}
