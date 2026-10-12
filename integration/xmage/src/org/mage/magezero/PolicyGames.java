package org.mage.magezero;

import mage.cards.decks.*;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.*;
import mage.constants.*;
import mage.game.*;
import mage.game.match.*;
import mage.game.mulligan.MulliganType;
import mage.player.ai.policy.PolicyPlayer;
import mage.players.Player;
import mage.util.RandomUtil;
import org.yaml.snakeyaml.Yaml;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Standalone direct-policy generator: deliberately bypasses the legacy MCTS/HDF5 runner.
 *
 * {@code PolicyGames game.yml} plays one batch and exits. {@code PolicyGames --serve}
 * boots once and then plays one batch per game.yml path read from stdin, answering
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
    private static final ScheduledExecutorService WATCHDOG = Executors.newSingleThreadScheduledExecutor(runnable -> {
        Thread thread = new Thread(runnable, "policy-game-watchdog");
        thread.setDaemon(true);
        return thread;
    });

    enum Outcome { TERMINAL, TRUNCATED, ERROR, TIMEOUT }

    public static void main(String[] args) throws Exception {
        if (args.length == 1 && "--serve".equals(args[0])) {
            serve();
            return;
        }
        if (args.length != 1) throw new IllegalArgumentException("usage: PolicyGames game.yml | PolicyGames --serve");
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
        Config.load(configPath);
        Map<String, Object> raw = rawConfig(configPath);
        String typeA = Config.INSTANCE.playerA.type, typeB = Config.INSTANCE.playerB.type;
        if (!"policy".equals(typeA) || !("policy".equals(typeB) || XMAGE_AI.equals(typeB))) {
            throw new IllegalArgumentException("player_a must be type: policy and player_b policy or " + XMAGE_AI);
        }
        if (!"normal".equals(Config.INSTANCE.gameMode)) {
            throw new IllegalArgumentException("v2 adapter currently supports two-player constructed games");
        }
        if (Config.INSTANCE.playerA.hiddenInfo.opponentHand || Config.INSTANCE.playerB.hiddenInfo.opponentHand) {
            throw new IllegalArgumentException("policy games require hiddenInfo.see_opponent_hand: false for both players");
        }
        if (Config.INSTANCE.playerA.gameplay.manualTap || Config.INSTANCE.playerB.gameplay.manualTap) {
            throw new IllegalArgumentException("initial policy adapter requires automatic engine mana payment");
        }
        if (Config.INSTANCE.training.threads < 1 || Config.INSTANCE.training.threads > 16
                || Config.INSTANCE.training.games < 1 || Config.INSTANCE.training.maxMinutes < 1
                || Config.INSTANCE.training.maxTurns < 1) throw new IllegalArgumentException("invalid collection limits (threads 1..16)");
        // Optional policy-service shards: game i uses shard i mod n for both players.
        List<Integer> ports = shardPorts(raw, "ports", Config.INSTANCE.server.port);
        List<Integer> opponentPorts = shardPorts(raw, "opponent_ports", Config.INSTANCE.server.opponentPort);
        Map<String, Object> server = section(raw, "server");
        // Odd games swap the two controllers between seats, so a learner facing a
        // fixed opponent plays both decks of an asymmetric matchup.
        boolean alternate = Boolean.TRUE.equals(server.get("alternate_seats"));
        int aiSkill = ((Number) section(raw, "opponent_ai").getOrDefault("skill", 6)).intValue();
        if (aiSkill < 1 || aiSkill > 10) throw new IllegalArgumentException("opponent_ai.skill must be in 1..10");
        // Import lists once; instantiate independent card/deck objects in each game.
        DeckCardLists a = DeckImporter.importDeckFromFile(Config.INSTANCE.playerA.deckPath, true);
        DeckCardLists b = DeckImporter.importDeckFromFile(Config.INSTANCE.playerB.deckPath, true);
        ExecutorService workers = Executors.newFixedThreadPool(Config.INSTANCE.training.threads);
        List<Future<Outcome>> games = new ArrayList<>();
        for (int i = 0; i < Config.INSTANCE.training.games; i++) {
            final int gameNumber = i;
            final int port = ports.get(i % ports.size());
            final int opponentPort = opponentPorts.get(i % opponentPorts.size());
            final boolean swap = alternate && i % 2 == 1;
            games.add(workers.submit(() -> play(a, b, gameNumber, port, opponentPort, typeB, swap, aiSkill)));
        }
        workers.shutdown();
        int[] counts = new int[Outcome.values().length];
        // Each game interrupts itself after max_minutes. This deadline is only a
        // backstop for a game stuck outside policy calls (an engine loop).
        long rounds = (games.size() + Config.INSTANCE.training.threads - 1) / Config.INSTANCE.training.threads;
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(rounds * (Config.INSTANCE.training.maxMinutes + 1L));
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
            Map<String, Object> raw = new Yaml().load(reader);
            return raw == null ? Collections.emptyMap() : raw;
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

    /** XMage's own "Mad" AI, loaded by name so this module needs no compile-time dependency. */
    private static Player xmageAi(String name, int skill) throws ReflectiveOperationException {
        return (Player) Class.forName("mage.player.ai.ComputerPlayer7")
            .getConstructor(String.class, RangeOfInfluence.class, int.class)
            .newInstance(name, RangeOfInfluence.ONE, skill);
    }

    private static Outcome play(DeckCardLists listA, DeckCardLists listB, int index, int port, int opponentPort,
                                String opponentType, boolean swap, int aiSkill) throws Exception {
        RandomUtil.setSeed(ThreadLocalRandom.current().nextLong());
        Thread self = Thread.currentThread();
        ScheduledFuture<?> watchdog = WATCHDOG.schedule(self::interrupt, Config.INSTANCE.training.maxMinutes, TimeUnit.MINUTES);
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE,
            MulliganType.GAME_DEFAULT.getMulligan(0), 60, 20, 7);
        Match match = new TwoPlayerMatch(new MatchOptions("policy selfplay", "policy", false));
        String host = "http://" + Config.INSTANCE.server.host + ":";
        // The learner controller always uses `port`; swapping moves it to seat B.
        Player learner = new PolicyPlayer(swap ? "PlayerB" : "PlayerA", RangeOfInfluence.ONE, host + port);
        String opponentName = swap ? "PlayerA" : "PlayerB";
        Player opponent = XMAGE_AI.equals(opponentType) ? xmageAi(opponentName, aiSkill)
            : new PolicyPlayer(opponentName, RangeOfInfluence.ONE, host + opponentPort);
        Player a = swap ? opponent : learner, b = swap ? learner : opponent;
        Outcome outcome = Outcome.ERROR;
        try {
            load(game, match, a, Deck.load(listA, false, false, new HashMap<>()));
            load(game, match, b, Deck.load(listB, false, false, new HashMap<>()));
            if (a instanceof PolicyPlayer) ((PolicyPlayer) a).initializeKnownDeck(game, b.getId());
            if (b instanceof PolicyPlayer) ((PolicyPlayer) b).initializeKnownDeck(game, a.getId());
            GameOptions options = new GameOptions();
            options.testMode = true;
            options.stopOnTurn = Config.INSTANCE.training.maxTurns;
            options.stopAtStep = PhaseStep.END_TURN;
            game.setGameOptions(options);
            if ("player_a".equals(Config.INSTANCE.goesFirst)) game.setStartingPlayerId(a.getId());
            else if ("player_b".equals(Config.INSTANCE.goesFirst)) game.setStartingPlayerId(b.getId());
            // Decorrelated from seat swapping: each controller starts with each deck.
            else game.setStartingPlayerId((index / 2) % 2 == 0 ? a.getId() : b.getId());
            game.start(null);
            // XMage swallows rules errors and ends test-mode games without a
            // winner, so only an error-free game that reached the turn limit is
            // a truncation; anything else unfinished is an error.
            if (self.isInterrupted()) outcome = Outcome.TIMEOUT;
            else if (game.getTotalErrorsCount() > 0) outcome = Outcome.ERROR;
            else if (a.hasWon() || b.hasWon() || (a.hasLost() && b.hasLost())) outcome = Outcome.TERMINAL;
            else if (game.getTurnNum() >= Config.INSTANCE.training.maxTurns) outcome = Outcome.TRUNCATED;
        } catch (Throwable error) {
            outcome = self.isInterrupted() ? Outcome.TIMEOUT : Outcome.ERROR;
            if (outcome == Outcome.ERROR) {
                System.err.println("policy game " + index + " failed:");
                error.printStackTrace();
            }
        }
        // Every Throwable is caught above, so this cleanup always runs.
        watchdog.cancel(false);
        Thread.interrupted();  // never leak an interrupt into the next pooled game
        if (outcome == Outcome.ERROR && game.getTotalErrorsCount() > 0) {
            System.err.println("policy game " + index + " ended after " + game.getTotalErrorsCount() + " rules-engine error(s)");
        }
        for (Player player : Arrays.asList(a, b)) {
            if (!(player instanceof PolicyPlayer)) continue;
            try {
                ((PolicyPlayer) player).finish(game, outcome == Outcome.TERMINAL, outcome == Outcome.TRUNCATED);
            } catch (RuntimeException error) {
                // A failed /finish discards this trajectory; it must not fail the batch.
                System.err.println("policy game " + index + " finish failed: " + error);
                if (outcome == Outcome.TERMINAL || outcome == Outcome.TRUNCATED) outcome = Outcome.ERROR;
            }
        }
        return outcome;
    }

    private static void load(Game game, Match match, Player player, Deck deck) {
        player.setTestMode(true);
        game.loadCards(deck.getCards(), player.getId());
        game.loadCards(deck.getSideboard(), player.getId());
        game.addPlayer(player, deck);
        match.addPlayer(player, deck);
    }
}
