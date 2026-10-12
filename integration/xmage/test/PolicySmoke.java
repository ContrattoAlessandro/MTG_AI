import mage.cards.CardImpl;
import mage.cards.CardSetInfo;
import mage.cards.decks.Deck;
import mage.constants.*;
import mage.game.*;
import mage.game.mulligan.MulliganType;
import mage.player.ai.policy.PolicyPlayer;
import java.util.UUID;

/** A real XMage rules game using synthetic cards; does not require Mage.Sets/DB. */
public final class PolicySmoke {
    private static final class SmokeCard extends CardImpl {
        SmokeCard(UUID owner, boolean creature) {
            super(owner, new CardSetInfo(creature ? "Card 1" : "Card 2", "TST", "1", Rarity.COMMON),
                new CardType[]{creature ? CardType.CREATURE : CardType.LAND}, "");
            if (creature) { power = new mage.MageInt(3); toughness = new mage.MageInt(3); }
        }
        private SmokeCard(SmokeCard other) { super(other); }
        @Override public SmokeCard copy() { return new SmokeCard(this); }
    }

    private static void load(Game game, PolicyPlayer player) {
        Deck deck = new Deck();
        for (int i = 0; i < 12; i++) deck.getCards().add(new SmokeCard(player.getId(), i % 3 != 0));
        player.setTestMode(true);
        game.loadCards(deck.getCards(), player.getId());
        game.addPlayer(player, deck);
    }

    public static void main(String[] args) {
        org.apache.log4j.BasicConfigurator.configure();
        org.apache.log4j.Logger.getRootLogger().setLevel(org.apache.log4j.Level.WARN);
        String endpoint = args.length == 0 ? "http://127.0.0.1:50054" : args[0];
        if (args.length > 1) {
            org.mage.magezero.Config.load(args[1]);
            if (org.mage.magezero.Config.INSTANCE.playerA.hiddenInfo.opponentHand
                    || org.mage.magezero.Config.INSTANCE.playerB.hiddenInfo.opponentHand) {
                throw new IllegalStateException("default v2 game configuration exposes private hands");
            }
        }
        Game game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE,
            MulliganType.GAME_DEFAULT.getMulligan(0), 0, 6, 3);
        PolicyPlayer a = new PolicyPlayer("Smoke A", RangeOfInfluence.ONE, endpoint);
        PolicyPlayer b = new PolicyPlayer("Smoke B", RangeOfInfluence.ONE, endpoint);
        load(game, a); load(game, b);
        a.initializeKnownDeck(game, b.getId()); b.initializeKnownDeck(game, a.getId());
        GameOptions options = new GameOptions();
        options.testMode = true; options.stopOnTurn = 20; options.stopAtStep = PhaseStep.END_TURN;
        game.setGameOptions(options); game.setStartingPlayerId(a.getId());
        boolean terminal = false;
        try {
            game.start(null);
            terminal = a.hasWon() || b.hasWon() || (a.hasLost() && b.hasLost());
            if (!terminal) throw new IllegalStateException("smoke game did not terminate");
            System.out.println("XMage smoke game finished: A=" + a.hasWon() + ", B=" + b.hasWon());
        } finally {
            a.finish(game, terminal, false); b.finish(game, terminal, false);
        }
    }
}
