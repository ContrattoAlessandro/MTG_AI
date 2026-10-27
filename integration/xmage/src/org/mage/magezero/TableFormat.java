package org.mage.magezero;

import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameCommanderImpl;
import mage.game.match.Match;
import mage.game.match.MatchOptions;
import mage.game.mulligan.Mulligan;

import java.util.Locale;

/**
 * The kinds of game a table (`mz lan`) can play, created as XMage's own tables create them.
 * The game classes live in XMage's game plugins and load by name. A new format is a new
 * constant here; nothing else changes.
 */
enum TableFormat {
    /** Two players, 20 life. */
    DUEL("mage.game.TwoPlayerDuel", "mage.game.TwoPlayerMatch", 2, 2, 20, false),
    /** Two Commander decks: 40 life, 21 commander damage loses. */
    COMMANDER_DUEL("mage.game.CommanderDuel", "mage.game.CommanderDuelMatch", 2, 2, 40, true),
    /** Two to ten players, each against all the others, 20 life. */
    FREE_FOR_ALL("mage.game.FreeForAll", "mage.game.FreeForAllMatch", 2, 10, 20, false),
    /** Multiplayer Commander: 40 life, 21 damage from one commander loses. */
    COMMANDER_FREE_FOR_ALL("mage.game.CommanderFreeForAll", "mage.game.CommanderFreeForAllMatch", 2, 10, 40, true);

    private static final int HAND_SIZE = 7;
    private final String gameClass, matchClass;
    final int minPlayers, maxPlayers, startingLife;
    final boolean commander;

    TableFormat(String gameClass, String matchClass, int minPlayers, int maxPlayers, int startingLife, boolean commander) {
        this.gameClass = gameClass;
        this.matchClass = matchClass;
        this.minPlayers = minPlayers;
        this.maxPlayers = maxPlayers;
        this.startingLife = startingLife;
        this.commander = commander;
    }

    /** "free-for-all" names FREE_FOR_ALL. */
    static TableFormat named(String name) {
        try {
            return valueOf(name.trim().toUpperCase(Locale.ROOT).replace('-', '_'));
        } catch (IllegalArgumentException error) {
            throw new IllegalArgumentException("unknown table format " + name
                + " (duel, commander-duel, free-for-all, commander-free-for-all)");
        }
    }

    boolean multiplayer() { return maxPlayers > 2; }

    /** Every player in range of every other; a creature may attack any opponent. */
    RangeOfInfluence range() { return multiplayer() ? RangeOfInfluence.ALL : RangeOfInfluence.ONE; }

    Game newGame(Mulligan mulligan) throws ReflectiveOperationException {
        MultiplayerAttackOption attack = multiplayer() ? MultiplayerAttackOption.MULTIPLE : MultiplayerAttackOption.LEFT;
        Game game;
        if (this == DUEL) {
            // TwoPlayerDuel also takes the minimum deck size.
            game = (Game) Class.forName(gameClass)
                .getConstructor(MultiplayerAttackOption.class, RangeOfInfluence.class, Mulligan.class, int.class, int.class, int.class)
                .newInstance(attack, range(), mulligan, 60, startingLife, HAND_SIZE);
        } else {
            game = (Game) Class.forName(gameClass)
                .getConstructor(MultiplayerAttackOption.class, RangeOfInfluence.class, Mulligan.class, int.class, int.class)
                .newInstance(attack, range(), mulligan, startingLife, HAND_SIZE);
        }
        if (commander) ((GameCommanderImpl) game).setCheckCommanderDamage(true);
        return game;
    }

    Match newMatch(MatchOptions options) throws ReflectiveOperationException {
        return (Match) Class.forName(matchClass).getConstructor(MatchOptions.class).newInstance(options);
    }
}
