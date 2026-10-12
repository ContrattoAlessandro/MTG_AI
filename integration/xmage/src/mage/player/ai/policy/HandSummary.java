package mage.player.ai.policy;

import mage.Mana;
import mage.abilities.mana.ActivatedManaAbilityImpl;
import mage.cards.Card;
import mage.constants.Zone;
import mage.game.Game;
import java.util.*;

/**
 * Opening-hand statistics for mulligan candidates (action slots 3-13). Colour fit is
 * an approximation: a spell fits when, for each colour, at least as many lands can
 * produce that colour as the spell has pips of it, and its mana value is covered.
 */
final class HandSummary {
    private HandSummary() {}

    /** Counts for a concrete hand, written to slots 3-13 (divided by 7). */
    static void describe(List<Card> hand, int size, double[] values, Game game) {
        List<Set<Integer>> sources = new ArrayList<>();
        List<Card> spells = new ArrayList<>();
        for (Card card : hand) {
            if (card.isLand(game)) sources.add(colours(card, game)); else spells.add(card);
        }
        int fit = 0, turnTwo = 0, turnThree = 0;
        for (Card spell : spells) {
            if (!coloursFit(spell, sources)) continue;
            fit++;
            int cost = spell.getManaValue();
            if (cost <= Math.min(sources.size(), 2)) turnTwo++;
            if (cost <= Math.min(sources.size(), 3)) turnThree++;
        }
        values[3] = size / 7.0;
        values[4] = sources.size() / 7.0;
        values[5] = spells.size() / 7.0;
        values[6] = fit / 7.0;
        values[7] = turnTwo / 7.0;
        values[8] = turnThree / 7.0;
        double[] curve = curve(spells);
        for (int i = 0; i < 5; i++) values[9 + i] = curve[i] / 7.0;
    }

    /** Expected counts in a uniformly drawn hand of `size` cards from `deck`. */
    static void expected(List<Card> deck, int size, double[] values, Game game) {
        values[3] = size / 7.0;
        if (deck.isEmpty()) return;
        List<Card> spells = new ArrayList<>();
        for (Card card : deck) if (!card.isLand(game)) spells.add(card);
        double scale = size / (double) deck.size();
        values[4] = (deck.size() - spells.size()) * scale / 7.0;
        values[5] = spells.size() * scale / 7.0;
        double[] curve = curve(spells);
        for (int i = 0; i < 5; i++) values[9 + i] = curve[i] * scale / 7.0;
    }

    /** Nonland mana-value buckets 0-1, 2, 3, 4, 5+. */
    private static double[] curve(List<Card> spells) {
        double[] buckets = new double[5];
        for (Card spell : spells) buckets[Math.max(0, Math.min(4, spell.getManaValue() - 1))]++;
        return buckets;
    }

    /** Colour indices 0-4 (W, U, B, R, G) a land's mana abilities can produce. */
    private static Set<Integer> colours(Card land, Game game) {
        Set<Integer> result = new HashSet<>();
        for (ActivatedManaAbilityImpl ability : land.getAbilities(game).getActivatedManaAbilities(Zone.BATTLEFIELD)) {
            for (Mana mana : ability.getNetMana(game)) {
                if (mana.getAny() > 0) { for (int i = 0; i < 5; i++) result.add(i); continue; }
                int[] amounts = {mana.getWhite(), mana.getBlue(), mana.getBlack(), mana.getRed(), mana.getGreen()};
                for (int i = 0; i < 5; i++) if (amounts[i] > 0) result.add(i);
            }
        }
        return result;
    }

    private static boolean coloursFit(Card spell, List<Set<Integer>> sources) {
        Mana cost = spell.getManaCost().getMana();
        int[] pips = {cost.getWhite(), cost.getBlue(), cost.getBlack(), cost.getRed(), cost.getGreen()};
        for (int colour = 0; colour < 5; colour++) {
            if (pips[colour] == 0) continue;
            int producers = 0;
            for (Set<Integer> source : sources) if (source.contains(colour)) producers++;
            if (producers < pips[colour]) return false;
        }
        return true;
    }
}
