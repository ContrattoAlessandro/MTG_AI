package org.mage.magezero;

import mage.cards.repository.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.text.Normalizer;
import java.util.*;

/**
 * Converts a plain "count name" deck list into XMage's .dck format.
 *
 * Lines after the first blank line are the sideboard; for a Commander deck that
 * section holds the commander, which XMage moves to the command zone. Every name
 * must resolve in the local card database, or the conversion fails and lists the
 * missing cards.
 */
public final class DeckListConverter {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("usage: DeckListConverter list.txt deck.dck");
        RepositoryUtil.bootstrapLocalDb();
        CardScanner.scan();
        List<String> output = new ArrayList<>();
        List<String> missing = new ArrayList<>();
        boolean sideboard = false;
        int main = 0, side = 0;
        for (String raw : Files.readAllLines(Paths.get(args[0]), StandardCharsets.UTF_8)) {
            String line = raw.replace("﻿", "").trim();
            if (line.isEmpty()) {
                if (main > 0) sideboard = true;
                continue;
            }
            int space = line.indexOf(' ');
            int count = Integer.parseInt(line.substring(0, space));
            String name = line.substring(space + 1).trim();
            CardInfo card = find(name);
            if (card == null) {
                missing.add(name);
                continue;
            }
            output.add((sideboard ? "SB: " : "") + count + " [" + card.getSetCode() + ":" + card.getCardNumber() + "] " + card.getName());
            if (sideboard) side += count; else main += count;
        }
        if (!missing.isEmpty()) {
            throw new IllegalStateException("cards not in the XMage database: " + missing);
        }
        Files.write(Paths.get(args[1]), output, StandardCharsets.UTF_8);
        System.out.println("Wrote " + args[1] + ": " + main + " main, " + side + " sideboard");
    }

    private static CardInfo find(String name) {
        CardInfo card = CardRepository.instance.findPreferredCoreExpansionCard(name, "");
        if (card != null) return card;
        // Card databases differ on accents (Éomer / Eomer).
        String ascii = Normalizer.normalize(name, Normalizer.Form.NFD).replaceAll("\\p{M}", "");
        return ascii.equals(name) ? null : CardRepository.instance.findPreferredCoreExpansionCard(ascii, "");
    }
}
