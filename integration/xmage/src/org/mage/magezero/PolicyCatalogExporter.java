package org.mage.magezero;

import com.google.gson.GsonBuilder;
import mage.MageObject;
import mage.abilities.Ability;
import mage.abilities.Mode;
import mage.abilities.effects.Effect;
import mage.cards.Card;
import mage.cards.ModalDoubleFacedCard;
import mage.cards.decks.Deck;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.*;
import mage.constants.CardType;
import mage.game.permanent.token.Token;
import java.lang.reflect.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.*;

/**
 * Export fixed IDs and card semantics for the complete intended training deck pool.
 *
 * Tokens the pool's cards can create are added under their token names (as
 * {@code object.getName()} reports them on the battlefield), so the policy sees a
 * Goblin or a Treasure rather than an anonymous token. Tokens are found by walking
 * the cards' effects; with {@code --sources Mage.Sets/src}, the card sources are
 * also scanned for tokens built inside custom effects.
 */
public final class PolicyCatalogExporter {
    private static final Pattern TOKEN_USE = Pattern.compile("new\\s+(\\w+Token)\\s*\\(|\\b(\\w+Token)::new");

    public static void main(String[] args) throws Exception {
        List<String> decks = new ArrayList<>(Arrays.asList(args));
        Path sources = null;
        int flag = decks.indexOf("--sources");
        if (flag >= 0) {
            if (flag + 1 >= decks.size()) throw new IllegalArgumentException("--sources needs a directory");
            sources = Paths.get(decks.remove(flag + 1));
            decks.remove(flag);
        }
        if (decks.size() < 2) {
            throw new IllegalArgumentException("usage: PolicyCatalogExporter catalog.json [--sources Mage.Sets/src] deck1.dck [deck2.dck ...]");
        }
        Path output = Paths.get(decks.remove(0));
        RepositoryUtil.bootstrapLocalDb();
        CardScanner.scan();
        Map<String, Card> deckCards = new TreeMap<>();
        for (String path : decks) {
            Deck deck = Deck.load(DeckImporter.importDeckFromFile(path, true), false, false, new HashMap<>());
            for (Card card : deck.getCards()) deckCards.put(card.getName(), card);
            for (Card card : deck.getSideboard()) deckCards.put(card.getName(), card);
        }
        Map<String, MageObject> pool = new TreeMap<>(deckCards);
        // Other faces (a transformed Ajani, an MDFC played as its land) appear on the
        // battlefield and stack under their own names.
        for (Card card : deckCards.values()) {
            List<Card> faces = new ArrayList<>();
            if (card instanceof ModalDoubleFacedCard) {
                faces.add(((ModalDoubleFacedCard) card).getLeftHalfCard());
                faces.add(((ModalDoubleFacedCard) card).getRightHalfCard());
            } else if (card.getSecondCardFace() != null) {
                faces.add(card.getSecondCardFace());
            }
            for (Card face : faces) pool.putIfAbsent(face.getName(), face);
        }
        Set<String> tokenNames = new TreeSet<>();
        for (Card card : deckCards.values()) {
            List<Token> tokens = new ArrayList<>();
            for (Ability ability : card.getAbilities()) collect(ability, tokens, 0);
            if (sources != null) tokens.addAll(sourceTokens(sources, card));
            for (Token token : tokens) {
                // A token copy of a card uses the card's ID; an unnamed template is a copy placeholder.
                if (deckCards.containsKey(token.getName()) || token.getName().replace("Token", "").trim().isEmpty()) continue;
                pool.putIfAbsent(token.getName(), token);
                tokenNames.add(token.getName());
            }
        }
        List<Map<String, Object>> cards = new ArrayList<>();
        int id = 1;
        for (Map.Entry<String, MageObject> entry : pool.entrySet()) {
            Map<String, Object> card = new LinkedHashMap<>();
            card.put("id", id++);
            card.put("key", (tokenNames.contains(entry.getKey()) ? "xmage-token:" : "xmage-name:") + entry.getKey());
            card.put("name", entry.getKey());
            card.put("features", features(entry.getValue()));
            cards.add(card);
        }
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("format_version", 1);
        catalog.put("cards", cards);
        if (output.getParent() != null) Files.createDirectories(output.getParent());
        Files.write(output, new GsonBuilder().setPrettyPrinting().create().toJson(catalog).getBytes(StandardCharsets.UTF_8));
        System.out.println("Exported " + cards.size() + " card kinds, " + tokenNames.size() + " of them tokens "
            + tokenNames + ", to " + output);
    }

    /** Tokens held by an ability's effects, and by abilities those effects grant; depth-limited. */
    private static void collect(Object value, List<Token> tokens, int depth) {
        if (value == null || depth > 4) return;
        if (value instanceof Token) {
            tokens.add((Token) value);
        } else if (value instanceof Ability) {
            for (Mode mode : ((Ability) value).getModes().values()) {
                for (Effect effect : mode.getEffects()) collect(effect, tokens, depth + 1);
            }
        } else if (value instanceof Collection) {
            for (Object item : (Collection<?>) value) collect(item, tokens, depth + 1);
        } else if (value instanceof Effect) {
            for (Class<?> type = value.getClass(); type != null && type != Object.class; type = type.getSuperclass()) {
                for (Field field : type.getDeclaredFields()) {
                    if (Modifier.isStatic(field.getModifiers()) || field.getType().isPrimitive()) continue;
                    try {
                        field.setAccessible(true);
                        collect(field.get(value), tokens, depth + 1);
                    } catch (ReflectiveOperationException | RuntimeException ignored) {
                        // inaccessible field: the source scan is the fallback
                    }
                }
            }
        }
    }

    /** Token classes named in the card's source; custom effects build their tokens inline. */
    private static List<Token> sourceTokens(Path root, Card card) throws Exception {
        Path file = root.resolve(card.getClass().getName().replace('.', '/') + ".java");
        if (!Files.exists(file)) {
            // Basic lands live in the core module and create no tokens.
            if (!card.getClass().getName().startsWith("mage.cards.basiclands.")) System.err.println("warning: card source not found: " + file);
            return Collections.emptyList();
        }
        String source = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
        List<Token> tokens = new ArrayList<>();
        Matcher matcher = TOKEN_USE.matcher(source);
        while (matcher.find()) {
            String simple = matcher.group(1) != null ? matcher.group(1) : matcher.group(2);
            Token token = instantiate(source, simple);
            if (token == null) System.err.println("warning: " + card.getName() + " uses " + simple + ", which could not be instantiated");
            else tokens.add(token);
        }
        return tokens;
    }

    private static Token instantiate(String source, String simple) {
        List<String> names = new ArrayList<>();
        Matcher imported = Pattern.compile("import\\s+([\\w.]+\\." + simple + ")\\s*;").matcher(source);
        if (imported.find()) names.add(imported.group(1));
        names.add("mage.game.permanent.token." + simple);
        names.add("mage.game.permanent.token.custom." + simple);
        for (String name : names) {
            try {
                Class<?> type = Class.forName(name);
                if (!Token.class.isAssignableFrom(type)) continue;
                for (Constructor<?> constructor : type.getConstructors()) {
                    // No-argument tokens, or X/X tokens built from integers (the name is fixed).
                    Class<?>[] parameters = constructor.getParameterTypes();
                    if (Arrays.stream(parameters).allMatch(parameter -> parameter == int.class)) {
                        Object[] arguments = new Object[parameters.length];
                        Arrays.fill(arguments, 1);
                        return (Token) constructor.newInstance(arguments);
                    }
                }
            } catch (ReflectiveOperationException | RuntimeException | LinkageError ignored) {
                // try the next package
            }
        }
        return null;
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
