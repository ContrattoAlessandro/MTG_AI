package org.mage.magezero;

import com.google.gson.GsonBuilder;
import mage.MageObject;
import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.*;
import mage.constants.CardType;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Export fixed IDs and card semantics for the complete intended training deck pool. */
public final class PolicyCatalogExporter {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) throw new IllegalArgumentException("usage: PolicyCatalogExporter catalog.json deck1.dck [deck2.dck ...]");
        RepositoryUtil.bootstrapLocalDb();
        CardScanner.scan();
        Map<String, Card> pool = new TreeMap<>();
        for (int i = 1; i < args.length; i++) {
            Deck deck = Deck.load(DeckImporter.importDeckFromFile(args[i], true), false, false, new HashMap<>());
            for (Card card : deck.getCards()) pool.put(card.getName(), card);
            for (Card card : deck.getSideboard()) pool.put(card.getName(), card);
        }
        List<Map<String, Object>> cards = new ArrayList<>();
        int id = 1;
        for (Map.Entry<String, Card> entry : pool.entrySet()) {
            Map<String, Object> card = new LinkedHashMap<>();
            card.put("id", id++);
            card.put("key", "xmage-name:" + entry.getKey());
            card.put("name", entry.getKey());
            card.put("features", features(entry.getValue()));
            cards.add(card);
        }
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("format_version", 1);
        catalog.put("cards", cards);
        Path output = Paths.get(args[0]);
        if (output.getParent() != null) Files.createDirectories(output.getParent());
        Files.write(output, new GsonBuilder().setPrettyPrinting().create().toJson(catalog).getBytes(StandardCharsets.UTF_8));
        System.out.println("Exported " + cards.size() + " card kinds to " + output);
    }

    private static double[] features(MageObject card) {
        double[] values = new double[32];
        CardType[] types = {CardType.LAND, CardType.CREATURE, CardType.INSTANT, CardType.SORCERY,
                           CardType.ARTIFACT, CardType.ENCHANTMENT, CardType.PLANESWALKER, CardType.BATTLE};
        for (int i = 0; i < types.length; i++) values[i] = card.getCardType().contains(types[i]) ? 1 : 0;
        values[8] = card.getManaValue() / 16.0;
        values[9] = card.getColor().isWhite() ? 1 : 0;
        values[10] = card.getColor().isBlue() ? 1 : 0;
        values[11] = card.getColor().isBlack() ? 1 : 0;
        values[12] = card.getColor().isRed() ? 1 : 0;
        values[13] = card.getColor().isGreen() ? 1 : 0;
        values[14] = card.getPower().getValue() / 20.0;
        values[15] = card.getToughness().getValue() / 20.0;
        String rules = card.getAbilities().toString().toLowerCase(Locale.ROOT);
        String[] keywords = {"flying", "first strike", "double strike", "deathtouch", "lifelink", "trample",
            "vigilance", "haste", "hexproof", "ward", "indestructible", "reach", "flash", "menace", "defender", "protection"};
        for (int i = 0; i < keywords.length; i++) values[16 + i] = rules.contains(keywords[i]) ? 1 : 0;
        return values;
    }
}
