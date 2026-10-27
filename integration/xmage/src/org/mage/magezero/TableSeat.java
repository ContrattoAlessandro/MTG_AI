package org.mage.magezero;

import mage.constants.RangeOfInfluence;
import mage.player.ai.XMageAiPlayer;
import mage.player.ai.policy.WebPlayer;
import mage.game.Game;
import mage.players.Player;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One seat of a table game file: who sits there and with which deck.
 *
 *   {"name": "Ann", "type": "web", "deckPath": "decks/winota.dck", "endpoint": "http://127.0.0.1:8765/engine/seat/1"}
 *   {"name": "Bot", "type": "xmage-ai", "deckPath": "decks/kinnan.dck", "skill": 4}
 *
 * Each seat type is a {@link Kind}: it makes the player and readies it once the game exists.
 */
final class TableSeat {
    /** How a seat type makes and readies its player. */
    interface Kind {
        Player create(TableSeat seat, RangeOfInfluence range);

        /** Called after every player has joined the game, before it starts. */
        default void ready(Player player, Game game) {}
    }

    private static final Map<String, Kind> KINDS = new LinkedHashMap<>();
    static {
        // A person in a browser: the web server at `endpoint` relays the decisions.
        KINDS.put(PolicyGames.WEB, new Kind() {
            @Override public Player create(TableSeat seat, RangeOfInfluence range) {
                if (seat.endpoint == null) throw new IllegalArgumentException("web seat " + seat.name + " needs an endpoint");
                return new WebPlayer(seat.name, range, null, seat.endpoint);
            }

            @Override public void ready(Player player, Game game) { ((WebPlayer) player).joinTable(game); }
        });
        // XMage's own AI, which plays games of any size.
        KINDS.put(PolicyGames.XMAGE_AI, (seat, range) -> {
            XMageAiPlayer player = new XMageAiPlayer(seat.name, range, seat.skill);
            if (seat.thinkSeconds > 0) player.setMaxThinkTimeSecs(seat.thinkSeconds);
            return player;
        });
    }

    final String name, type, deckPath, endpoint;
    final int skill, thinkSeconds;

    TableSeat(Map<String, Object> raw) {
        name = text(raw, "name", null);
        type = text(raw, "type", "");
        deckPath = text(raw, "deckPath", null);
        endpoint = text(raw, "endpoint", null);
        skill = number(raw, "skill", 4);
        thinkSeconds = number(raw, "think_seconds", 0);
        if (name == null || name.trim().isEmpty()) throw new IllegalArgumentException("every seat needs a name");
        if (deckPath == null) throw new IllegalArgumentException("seat " + name + " needs a deckPath");
        if (!KINDS.containsKey(type)) throw new IllegalArgumentException("seat " + name + ": type must be one of " + KINDS.keySet());
        if (skill < 1 || skill > 10) throw new IllegalArgumentException("seat " + name + ": skill must be in 1..10");
    }

    Kind kind() { return KINDS.get(type); }

    private static String text(Map<String, Object> raw, String key, String fallback) {
        Object value = raw.get(key);
        return value == null ? fallback : String.valueOf(value);
    }

    private static int number(Map<String, Object> raw, String key, int fallback) {
        Object value = raw.get(key);
        return value instanceof Number ? ((Number) value).intValue() : fallback;
    }
}
