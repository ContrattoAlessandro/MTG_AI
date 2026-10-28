package org.mage.magezero;

import mage.cards.decks.Deck;
import mage.cards.decks.importer.DeckImporter;
import mage.constants.PhaseStep;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.match.Match;
import mage.game.match.MatchOptions;
import mage.game.mulligan.MulliganType;
import mage.player.ai.policy.RecordingPlayer;
import mage.player.ai.policy.WebPlayer;
import mage.players.Player;
import mage.util.RandomUtil;
import mage.util.ThreadUtils;

import java.util.*;
import java.util.concurrent.*;

/**
 * One game at a table of two or more seats (`mz lan`): people in browsers (WebPlayer)
 * and XMage's AI, in any format of {@link TableFormat}. The game file:
 *
 *   {"format": "commander-free-for-all", "goes_first": "random" | seat index,
 *    "max_turns": 200, "max_minutes": 720, "seats": [ ...TableSeat... ]}
 *
 * Returns the batch counts PolicyGames reports: {terminal, requested, truncated, errors, timeouts}.
 */
final class TableGame {
    private TableGame() {}

    static int[] play(Map<String, Object> raw) throws Exception {
        TableFormat format = TableFormat.named(String.valueOf(raw.getOrDefault("format", "free-for-all")));
        List<TableSeat> seats = seats(raw.get("seats"));
        if (seats.size() < format.minPlayers || seats.size() > format.maxPlayers) {
            throw new IllegalArgumentException(format + " is played by " + format.minPlayers + " to " + format.maxPlayers + " players, not " + seats.size());
        }
        int maxTurns = number(raw, "max_turns", 200), maxMinutes = number(raw, "max_minutes", 720);
        int first = startingSeat(raw.get("goes_first"), seats.size());
        // XMage runs game code only on threads it names as game threads (ThreadUtils).
        ExecutorService runner = Executors.newSingleThreadExecutor(
            runnable -> new Thread(runnable, ThreadUtils.THREAD_PREFIX_GAME + " table"));
        Future<PolicyGames.Outcome> game = runner.submit(() -> run(format, seats, first, maxTurns));
        runner.shutdown();
        PolicyGames.Outcome outcome;
        try {
            outcome = game.get(maxMinutes, TimeUnit.MINUTES);
        } catch (TimeoutException error) {
            game.cancel(true);
            outcome = PolicyGames.Outcome.TIMEOUT;
        } catch (ExecutionException error) {
            error.getCause().printStackTrace();
            outcome = PolicyGames.Outcome.ERROR;
        } finally {
            runner.shutdownNow();
        }
        return new int[] {outcome == PolicyGames.Outcome.TERMINAL ? 1 : 0, 1, outcome == PolicyGames.Outcome.TRUNCATED ? 1 : 0,
                          outcome == PolicyGames.Outcome.ERROR ? 1 : 0, outcome == PolicyGames.Outcome.TIMEOUT ? 1 : 0};
    }

    @SuppressWarnings("unchecked")
    private static List<TableSeat> seats(Object listed) {
        if (!(listed instanceof List)) throw new IllegalArgumentException("seats must be a list");
        List<TableSeat> seats = new ArrayList<>();
        Set<String> names = new HashSet<>();
        for (Object entry : (List<?>) listed) {
            if (!(entry instanceof Map)) throw new IllegalArgumentException("every seat must be an object");
            TableSeat seat = new TableSeat((Map<String, Object>) entry);
            if (!names.add(seat.name)) throw new IllegalArgumentException("two seats are named " + seat.name);
            seats.add(seat);
        }
        return seats;
    }

    /** The index of the seat that starts: the given one, or a random one. */
    private static int startingSeat(Object goesFirst, int count) {
        if (goesFirst instanceof Number) {
            int index = ((Number) goesFirst).intValue();
            if (index < 0 || index >= count) throw new IllegalArgumentException("goes_first must be a seat index or random");
            return index;
        }
        return ThreadLocalRandom.current().nextInt(count);
    }

    private static PolicyGames.Outcome run(TableFormat format, List<TableSeat> seats, int first, int maxTurns) throws Exception {
        RandomUtil.setSeed(ThreadLocalRandom.current().nextLong());
        Game game = format.newGame(MulliganType.GAME_DEFAULT.getMulligan(0));
        Match match = format.newMatch(new MatchOptions("mz lan", "table", seats.size() > 2));
        List<Player> players = new ArrayList<>();
        for (TableSeat seat : seats) players.add(seat.kind().create(seat, format.range()));
        // XMage's turn order puts each joining player first (PlayerList): the players join
        // last seat first, so the turns go round in the order of the seats.
        for (int i = seats.size() - 1; i >= 0; i--) {
            Deck deck = Deck.load(DeckImporter.importDeckFromFile(seats.get(i).deckPath, true), false, false, new HashMap<>());
            PolicyGames.load(game, match, players.get(i), deck);
        }
        for (int i = 0; i < seats.size(); i++) seats.get(i).kind().ready(seats.get(i), players.get(i), game);
        // When every person at the table is out, the bots do not play on alone.
        List<UUID> people = new ArrayList<>();
        for (int i = 0; i < seats.size(); i++) if (PolicyGames.WEB.equals(seats.get(i).type)) people.add(players.get(i).getId());
        boolean[] peopleOut = {false};
        if (!people.isEmpty()) game.addTableEventListener(event -> {
            if (peopleOut[0] || game.isSimulation()) return;
            for (UUID id : people) if (!game.getPlayer(id).hasLost()) return;
            peopleOut[0] = true;
            game.end(); // the game loop stops at its next check (setGameStopped is not one)
        });
        GameOptions options = new GameOptions();
        // A real game: XMage's test mode is for its unit tests and deals no opening hands.
        options.testMode = false;
        options.stopOnTurn = maxTurns;
        options.stopAtStep = PhaseStep.END_TURN;
        game.setGameOptions(options);
        game.setStartingPlayerId(players.get(first).getId());
        PolicyGames.Outcome outcome = PolicyGames.Outcome.ERROR;
        try {
            game.start(null);
            boolean decided = players.stream().anyMatch(Player::hasWon)
                || players.stream().filter(player -> !player.hasLost()).count() <= 1;
            if (Thread.currentThread().isInterrupted()) outcome = PolicyGames.Outcome.TIMEOUT;
            else if (game.getTotalErrorsCount() > 0) outcome = PolicyGames.Outcome.ERROR;
            else if (decided || peopleOut[0]) outcome = PolicyGames.Outcome.TERMINAL;
            else if (game.getTurnNum() >= maxTurns) outcome = PolicyGames.Outcome.TRUNCATED;
        } catch (Throwable error) {
            System.err.println("table game failed:");
            error.printStackTrace();
        } finally {
            WebPlayer.leaveTable(game);
        }
        Thread.interrupted();
        StringBuilder summary = new StringBuilder("table game " + outcome + ": turn " + game.getTurnNum());
        for (Player player : players) {
            summary.append("; ").append(player.getName()).append(player.hasWon() ? " won" : player.hasLost() ? " lost" : " -")
                .append(" life ").append(player.getLife());
        }
        System.err.println(summary);
        for (Player player : players) {
            if (!(player instanceof RecordingPlayer)) continue;
            try {
                ((RecordingPlayer) player).finish(game, outcome == PolicyGames.Outcome.TERMINAL, outcome == PolicyGames.Outcome.TRUNCATED);
            } catch (RuntimeException error) {
                System.err.println("table game: " + player.getName() + " finish failed: " + error);
            }
        }
        return outcome;
    }

    private static int number(Map<String, Object> raw, String key, int fallback) {
        Object value = raw.get(key);
        return value instanceof Number ? ((Number) value).intValue() : fallback;
    }
}
