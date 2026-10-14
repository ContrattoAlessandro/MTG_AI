package mage.player.ai.policy;

import mage.MageObject;
import mage.abilities.Abilities;
import mage.abilities.Ability;
import mage.abilities.keyword.*;
import mage.cards.*;
import mage.constants.*;
import mage.counters.CounterType;
import mage.counters.Counters;
import mage.game.*;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentToken;
import mage.game.stack.StackObject;
import mage.players.Player;
import mage.watchers.common.CommanderInfoWatcher;
import mage.watchers.common.CommanderPlaysCountWatcher;
import java.util.*;

/** Entity snapshots with a strict private/public split. The deck is known before play. */
public final class PolicyEncoder {
    private final UUID observer;
    private final UUID opponent;
    private final PolicyClient client;
    private final Map<Integer, Integer> deck = new TreeMap<>();
    private final Map<UUID, Integer> originalDeckCards = new HashMap<>();
    private final String gameId = UUID.randomUUID().toString();
    private PolicyHistory history;
    private int step = 0;
    private List<Integer> lastCandidateIds = new ArrayList<>();
    private Set<UUID> commanderIds = Collections.emptySet();
    private CommanderPlaysCountWatcher commanderPlays;

    /** Entity vector width; must match ENTITY_FEATURES in schema.py. */
    public static final int ENTITY_FEATURES = 48;

    public PolicyEncoder(UUID observer, UUID opponent, PolicyClient client, Game game) {
        this.observer = observer;
        this.opponent = opponent;
        this.client = client;
        // Called BEFORE game.start: enumerating the announced deck list cannot
        // disclose a shuffle, hand, library order, or later hidden game state.
        // A Commander game moves the sideboard (the commander) to the command
        // zone, so it is part of the known deck: public while there, and a
        // legitimate hand candidate if it is ever returned to hand.
        boolean commander = game instanceof GameCommanderImpl;
        for (Card card : game.getCards()) {
            if (!card.getOwnerId().equals(opponent)) continue;
            if (!commander && game.getPlayer(opponent).getSideboard().contains(card.getId())) continue;
            if (!card.getId().equals(card.getMainCard().getId())) continue;
            Integer id = client.cardIds.get(card.getName());
            if (id == null) throw new IllegalStateException("card catalog is missing known deck card: " + card.getName());
            deck.put(id, deck.getOrDefault(id, 0) + 1);
            originalDeckCards.put(card.getId(), id);
        }
        if (deck.isEmpty()) throw new IllegalStateException("opponent deck was not loaded before policy initialization");
    }

    public void attachHistory(Game game) {
        if (history == null) {
            history = new PolicyHistory(observer, client.cardIds);
            game.getState().addWatcher(history);
        }
        // XMage may restore a copied state during rules processing.
        history = (PolicyHistory) game.getState().getWatchers().get(history.getKey());
    }

    public static Map<String, Object> map(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) result.put((String) pairs[i], pairs[i + 1]);
        return result;
    }

    public static int zone(Zone zone) {
        if (zone == null) return 0;
        switch (zone) {
            case BATTLEFIELD: return 1;
            case HAND: return 2;
            case GRAVEYARD: return 3;
            case EXILED: return 4;
            case STACK: return 5;
            case LIBRARY: return 6;
            case COMMAND: return 7;
            default: return 0;
        }
    }

    // Entity slots for current keywords (docs/policy-protocol.md).
    private static final Class<?>[] KEYWORDS = {FlyingAbility.class, FirstStrikeAbility.class, DoubleStrikeAbility.class,
        DeathtouchAbility.class, LifelinkAbility.class, TrampleAbility.class, VigilanceAbility.class, HasteAbility.class,
        HexproofBaseAbility.class, IndestructibleAbility.class, ReachAbility.class, MenaceAbility.class};
    private static final int[] KEYWORD_SLOTS = {10, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30};

    private int side(UUID id) { return observer.equals(id) ? 0 : opponent.equals(id) ? 1 : 2; }

    private void addObject(Map<String, Map<String, Object>> entities, MageObject object,
                           int kind, int zone, UUID owner, UUID controller, String visibility, Game game) {
        if (object == null || entities.containsKey(object.getId().toString())) return;
        boolean hidden = visibility.equals("hidden");
        boolean concealedIdentity = hidden || (object instanceof Card && ((Card) object).isFaceDown(game));
        double[] values = new double[ENTITY_FEATURES];
        int cardId = concealedIdentity ? 0 : client.cardIds.getOrDefault(object.getName(), 0);
        if (!hidden) {
            values[0] = object.getPower().getValue() / 20.0;
            values[1] = object.getToughness().getValue() / 20.0;
            values[2] = object.getManaValue() / 16.0;
            values[21] = object.isCreature(game) ? 1 : 0;
            values[22] = object.isLand(game) ? 1 : 0;
            values[23] = object.isArtifact(game) ? 1 : 0;
            values[24] = object.isEnchantment(game) ? 1 : 0;
            values[25] = object.isPlaneswalker(game) ? 1 : 0;
            // Current keywords, including abilities gained during play.
            Abilities<Ability> abilities = object instanceof Card ? ((Card) object).getAbilities(game) : object.getAbilities();
            for (int i = 0; i < KEYWORDS.length; i++) {
                for (Ability ability : abilities) {
                    if (KEYWORDS[i].isInstance(ability)) { values[KEYWORD_SLOTS[i]] = 1; break; }
                }
            }
            values[13] = object instanceof PermanentToken ? 1 : 0;
            values[31] = object.isCopy() ? 1 : 0;
            if (object instanceof Permanent) {
                Permanent permanent = (Permanent) object;
                Counters counters = permanent.getCounters(game);
                int plus = counters.getCount(CounterType.P1P1), minus = counters.getCount(CounterType.M1M1);
                int loyalty = counters.getCount(CounterType.LOYALTY);
                values[3] = permanent.isTapped() ? 1 : 0;
                values[4] = permanent.hasSummoningSickness() ? 1 : 0;
                values[5] = permanent.getDamage() / 20.0;
                values[6] = (counters.getTotalCount() - plus - minus - loyalty) / 20.0;
                values[7] = plus / 20.0;
                values[11] = minus / 20.0;
                values[12] = loyalty / 20.0;
                values[8] = permanent.isAttacking() ? 1 : 0;
                values[9] = permanent.getBlocking() > 0 ? 1 : 0;
                values[26] = permanent.isTransformed() ? 1 : 0;
            }
            UUID identity = object instanceof StackObject ? ((StackObject) object).getSourceId() : object.getId();
            if (commanderIds.contains(identity)) {
                values[32] = 1;
                // Casts from the command zone; each adds {2} of commander tax.
                values[33] = commanderPlays == null ? 0 : commanderPlays.getPlaysCount(identity) / 8.0;
            }
        }
        entities.put(object.getId().toString(), map("id", object.getId().toString(), "card_id", cardId,
            "kind", kind, "zone", zone, "owner", side(owner), "controller", side(controller),
            "visibility", visibility, "features", values));
    }

    public Map<String, Object> observation(Game game, List<Map<String, Object>> actions, Collection<Card> visibleChoices) {
        attachHistory(game);
        Map<String, Map<String, Object>> entities = new LinkedHashMap<>();
        Set<UUID> publicIdentities = new HashSet<>();
        Player me = game.getPlayer(observer), enemy = game.getPlayer(opponent);
        commanderIds = new HashSet<>();
        for (Player player : Arrays.asList(me, enemy)) commanderIds.addAll(game.getCommandersIds(player, CommanderCardType.ANY, false));
        commanderPlays = game.getState().getWatcher(CommanderPlaysCountWatcher.class);
        for (Player player : Arrays.asList(me, enemy)) {
            double[] features = new double[ENTITY_FEATURES];
            features[11] = player.getLife() / 40.0;
            features[12] = player.getCountersCount(CounterType.POISON) / 10.0;
            features[13] = player.getCountersCount(CounterType.ENERGY) / 20.0;
            features[7] = player.getHand().size() / 60.0;
            features[10] = Math.min(player.getLandsPlayed(), 4) / 4.0;
            features[14] = Math.max(0, Math.min(player.getLandsPerTurn(), 4) - player.getLandsPlayed()) / 4.0;
            // Combat damage taken from the other player's commanders (21 loses).
            Player other = player == me ? enemy : me;
            int commanderDamage = 0;
            for (UUID commanderId : game.getCommandersIds(other, CommanderCardType.ANY, false)) {
                CommanderInfoWatcher watcher = game.getState().getWatcher(CommanderInfoWatcher.class, commanderId);
                if (watcher != null) commanderDamage = Math.max(commanderDamage, watcher.getDamageToPlayer().getOrDefault(player.getId(), 0));
            }
            features[15] = commanderDamage / 21.0;
            entities.put(player.getId().toString(), map("id", player.getId().toString(), "card_id", 0,
                "kind", 0, "zone", 0, "owner", side(player.getId()), "controller", side(player.getId()),
                "visibility", "public", "features", features));
            for (Card card : player.getGraveyard().getCards(game)) {
                addObject(entities, card, 2, 3, player.getId(), player.getId(), "public", game);
                publicIdentities.add(card.getId());
            }
            double[] count = new double[ENTITY_FEATURES];
            count[7] = player.getLibrary().size() / 100.0;
            entities.put("library-" + player.getId(), map("id", "library-" + player.getId(), "card_id", 0,
                "kind", 4, "zone", 6, "owner", side(player.getId()), "controller", side(player.getId()),
                "visibility", "hidden", "features", count));
            for (UUID commanderId : game.getCommandersIds(player, CommanderCardType.ANY, false)) {
                if (game.getState().getZone(commanderId) != Zone.COMMAND) continue;
                addObject(entities, game.getCard(commanderId), 2, 7, player.getId(), player.getId(), "public", game);
                publicIdentities.add(commanderId);
            }
        }
        for (Card card : me.getHand().getCards(game)) {
            addObject(entities, card, 2, 2, observer, observer, "own", game);
        }
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            boolean facedown = permanent.isFaceDown(game);
            addObject(entities, permanent, 1, 1, permanent.getOwnerId(), permanent.getControllerId(),
                      "public", game);
            if (!facedown) publicIdentities.add(permanent.getId());
        }
        for (StackObject stack : game.getStack()) {
            addObject(entities, stack, 3, 5, stack.getControllerId(), stack.getControllerId(), "public", game);
            publicIdentities.add(stack.getSourceId());
        }
        for (ExileZone exile : game.getExile().getExileZones()) {
            for (Card card : exile.getCards(game)) {
                if (!card.isFaceDown(game)) {
                    addObject(entities, card, 2, 4, card.getOwnerId(), card.getOwnerId(), "public", game);
                    publicIdentities.add(card.getId());
                }
            }
        }
        // Explicit choice sets are cards XMage permits the observing player to
        // inspect (e.g. own-library searches). Never enumerate an opponent hand.
        if (visibleChoices != null) {
            for (Card card : visibleChoices) {
                Zone zone = game.getState().getZone(card.getId());
                addObject(entities, card, 2, zone(zone), card.getOwnerId(), observer, "public", game);
                // A choice that shows opponent hand cards (Thoughtseize-style) makes them
                // known, so the hand belief's lower bounds agree with the visible entities.
                if (zone == Zone.HAND && !observer.equals(card.getOwnerId()) && !card.isFaceDown(game)) {
                    history.revealedInHand(card.getId(), client.cardIds.getOrDefault(card.getName(), 0));
                }
            }
        }
        List<Map<String, Object>> relations = new ArrayList<>();
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            for (UUID attachment : permanent.getAttachments()) {
                if (entities.containsKey(attachment.toString())) relations.add(map("source", attachment.toString(),
                    "target", permanent.getId().toString(), "type", 0));
            }
        }
        for (StackObject stack : game.getStack()) {
            Ability ability = stack.getStackAbility();
            if (ability == null) continue;
            for (mage.target.Target target : ability.getTargets()) {
                for (UUID targetId : target.getTargets()) {
                    if (entities.containsKey(targetId.toString())) relations.add(map("source", stack.getId().toString(),
                        "target", targetId.toString(), "type", 1));
                }
            }
        }
        for (mage.game.combat.CombatGroup group : game.getCombat().getGroups()) {
            for (UUID attacker : group.getAttackers()) {
                UUID defender = group.getDefenderId();
                if (defender != null && entities.containsKey(attacker.toString()) && entities.containsKey(defender.toString())) {
                    relations.add(map("source", attacker.toString(), "target", defender.toString(), "type", 2));
                }
                for (UUID blocker : group.getBlockers()) {
                    if (entities.containsKey(blocker.toString()) && entities.containsKey(attacker.toString())) {
                        relations.add(map("source", blocker.toString(), "target", attacker.toString(), "type", 3));
                    }
                }
            }
        }
        // References missing from the visible snapshot are intentionally null.
        // The actor never learns a hidden object identity from a candidate pointer.
        for (Map<String, Object> action : actions) {
            for (String field : Arrays.asList("source", "target")) {
                Object reference = action.get(field);
                if (reference != null && !entities.containsKey(reference.toString())) action.put(field, null);
            }
        }
        Map<Integer, Integer> upper = new TreeMap<>(deck);
        for (UUID id : publicIdentities) {
            Integer cardId = originalDeckCards.get(id);
            if (cardId != null) upper.put(cardId, Math.max(0, upper.get(cardId) - 1));
        }
        List<Map<String, Object>> candidates = new ArrayList<>();
        lastCandidateIds = new ArrayList<>(deck.keySet());
        Map<Integer, Integer> known = history.knownHandCounts();
        for (Integer id : lastCandidateIds) candidates.add(map("card_id", id, "lower", known.getOrDefault(id, 0), "upper", upper.get(id)));
        double[] global = new double[24];
        global[0] = game.getTurnNum() / 100.0;
        global[1] = observer.equals(game.getActivePlayerId()) ? 1 : 0;
        global[2] = game.getStack().size() / 16.0;
        global[3] = me.getLife() / 40.0;
        global[4] = enemy.getLife() / 40.0;
        global[5] = me.getHand().size() / 60.0;
        global[6] = enemy.getHand().size() / 60.0;
        global[7] = me.getLibrary().size() / 100.0;
        global[8] = enemy.getLibrary().size() / 100.0;
        global[9] = game.getTurnStepType() == null ? 0 : game.getTurnStepType().ordinal() / 32.0;
        int[] myMana = {me.getManaPool().getWhite(), me.getManaPool().getBlue(), me.getManaPool().getBlack(),
            me.getManaPool().getRed(), me.getManaPool().getGreen(), me.getManaPool().getColorless()};
        int[] enemyMana = {enemy.getManaPool().getWhite(), enemy.getManaPool().getBlue(), enemy.getManaPool().getBlack(),
            enemy.getManaPool().getRed(), enemy.getManaPool().getGreen(), enemy.getManaPool().getColorless()};
        for (int i = 0; i < 6; i++) { global[10 + i] = myMana[i] / 20.0; global[16 + i] = enemyMana[i] / 20.0; }
        return map("schema_version", 2, "catalog_digest", client.catalogDigest,
            "game_id", gameId, "player_id", observer.toString(), "step", step++,
            "entities", new ArrayList<>(entities.values()), "relations", relations,
            "globals", global, "events", history.drain(), "actions", actions,
            "opponent_deck", candidates, "opponent_hand_size", enemy.getHand().size());
    }

    /** Privileged targets. Call separately, and send them only after the game. */
    public Map<String, Object> trainingLabel(Game game) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (Card card : game.getPlayer(opponent).getHand().getCards(game)) {
            Integer id = client.cardIds.get(card.getName());
            if (id == null || !deck.containsKey(id)) {
                throw new IllegalStateException("generated/copied off-deck hand card needs an extended candidate protocol: " + card.getName());
            }
            counts.put(id, counts.getOrDefault(id, 0) + 1);
        }
        List<Integer> labels = new ArrayList<>();
        for (Integer id : lastCandidateIds) labels.add(counts.getOrDefault(id, 0));
        return map("opponent_hand_counts", labels);
    }

    public String gameId() { return gameId; }
    public int decisions() { return step; }
    public void recordChoice(Map<String, Object> chosen, Game game) {
        attachHistory(game);
        history.choice((Integer) chosen.get("card_id"), (double[]) chosen.get("features"), game);
        // Own passes only. Writing them into the opponent policy's history too gave
        // it pass events from policy opponents but never from XMage's AI or humans,
        // so self-play and gauntlet/human games would have different histories.
        if ("pass".equals(chosen.get("type"))) history.passed(observer, game);
    }
}
