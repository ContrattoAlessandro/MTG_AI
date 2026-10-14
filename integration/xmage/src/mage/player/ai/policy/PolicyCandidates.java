package mage.player.ai.policy;

import mage.MageObject;
import mage.abilities.*;
import mage.abilities.common.PassAbility;
import mage.cards.Card;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.game.Game;
import mage.game.combat.CombatGroup;
import mage.game.permanent.Permanent;
import mage.players.Player;
import mage.target.Target;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Legal-action candidates for one decision. The policy player and the XMage-AI
 * teacher both build their lists here, so a teacher label indexes exactly the list
 * (order, ids and features) the policy is offered in the same state.
 *
 * options.get(i) is the engine object behind actions.get(i); null stands for the
 * pass/stop/stay/none option of the decisions that have one.
 */
public final class PolicyCandidates<T> {
    /** The policy service's max_actions. */
    public static final int MAX_OPTIONS = 2048;

    public final List<T> options = new ArrayList<>();
    public final List<Map<String, Object>> actions = new ArrayList<>();
    /** Cards the observer may inspect for this choice, or null. */
    public Collection<Card> visible;

    private PolicyCandidates<T> add(T option, Map<String, Object> action) {
        options.add(option);
        actions.add(action);
        return this;
    }

    public int size() { return actions.size(); }

    // Positions in UUID-sorted candidate lists (priority, targets, triggers,
    // replacement effects) differ every game, so they are never features.
    public static Map<String, Object> action(PolicyClient client, String id, String type, UUID source, UUID target, Game game) {
        double[] features = new double[16];
        MageObject object = source == null ? null : game.getObject(source);
        boolean concealed = object instanceof Card && ((Card) object).isFaceDown(game);
        int cardId = object == null || client == null || concealed ? 0 : client.cardIds.getOrDefault(object.getName(), 0);
        return PolicyEncoder.map("id", id, "type", type,
            "source", source == null ? null : source.toString(), "target", target == null ? null : target.toString(),
            "card_id", cardId, "features", features);
    }

    /** Slot 2: position in a list whose order is the same in every game (no/yes, card modes, sorted names). */
    public static Map<String, Object> ordinal(Map<String, Object> action, int index) {
        ((double[]) action.get("features"))[2] = index / 64.0;
        return action;
    }

    public static Map<String, Object> describe(Map<String, Object> action, String text) {
        // Candidate semantics supplement card/pointer embeddings. Stable lexical
        // bins distinguish modes and named choices across changing candidate sets.
        double[] values = (double[]) action.get("features");
        for (String word : String.valueOf(text).toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
            if (!word.isEmpty()) values[3 + Math.floorMod(word.hashCode(), 13)] += 1;
        }
        for (int i = 3; i < 16; i++) values[i] = Math.tanh(values[i] / 4.0);
        // Display text for a person choosing (WebPlayer); PolicyEncoder removes it
        // before an observation is sent, so it never reaches the policy.
        action.put("text", String.valueOf(text));
        return action;
    }

    public static String abilityKey(Ability ability) {
        return ability.getSourceId() + "|" + ability.getOriginalId();
    }

    /** Pass first, then playable non-mana abilities not in `rejected`. */
    public static PolicyCandidates<ActivatedAbility> priority(Player player, Game game, PolicyClient client, Set<String> rejected) {
        List<ActivatedAbility> abilities = new ArrayList<>(player.getPlayable(game, true));
        abilities.removeIf(Ability::isManaAbility);
        abilities.removeIf(ability -> rejected.contains(abilityKey(ability)));
        abilities.sort(Comparator.comparing((Ability a) -> String.valueOf(a.getSourceId())).thenComparing(Ability::toString));
        abilities.add(0, new PassAbility());
        PolicyCandidates<ActivatedAbility> result = new PolicyCandidates<>();
        for (int i = 0; i < abilities.size(); i++) {
            ActivatedAbility ability = abilities.get(i);
            String type = ability instanceof PassAbility ? "pass" : ability instanceof SpellAbility ? "cast" : "activate";
            Map<String, Object> candidate = action(client, "priority-" + i, type, ability.getSourceId(), null, game);
            ((double[]) candidate.get("features"))[1] = ability.getManaCosts().manaValue() / 16.0;
            result.add(i == 0 ? null : ability, describe(candidate, ability.getRule()));
        }
        return result;
    }

    /** One more target (or stop, once the target is legally chosen); empty when none remain. */
    public static PolicyCandidates<UUID> target(Player player, Target target, Ability source, Game game, Cards fromCards,
                                                PolicyClient client) {
        PolicyCandidates<UUID> result = new PolicyCandidates<>();
        UUID controller = target.getAffectedAbilityControllerId(player.getId());
        List<UUID> candidates = target.possibleTargets(controller, source, game, fromCards)
                .stream().filter(id -> !target.contains(id)).sorted().collect(Collectors.toList());
        if (candidates.isEmpty()) return result;
        // The target prompt (e.g. "put on the bottom of your library") gives the
        // choice its purpose; the pointer gives the object.
        String prompt = target.getMessage(game);
        UUID sourceId = source == null ? null : source.getSourceId();
        if (target.isChosen(game)) result.add(null, describe(action(client, "stop", "target", sourceId, null, game), prompt));
        for (int i = 0; i < candidates.size(); i++) {
            result.add(candidates.get(i), describe(action(client, "target-" + i, "target", sourceId, candidates.get(i), game), prompt));
        }
        result.visible = fromCards == null ? null : fromCards.getCards(game);
        return result;
    }

    public static PolicyCandidates<Boolean> use(String message, String trueText, String falseText, Ability source, Game game,
                                                PolicyClient client) {
        UUID reference = source == null ? null : source.getSourceId();
        return new PolicyCandidates<Boolean>()
            .add(false, ordinal(describe(action(client, "no", "use", reference, null, game), message + " " + falseText), 0))
            .add(true, ordinal(describe(action(client, "yes", "use", reference, null, game), message + " " + trueText), 1));
    }

    /**
     * Keep describes the current hand; mulligan describes an average redraw of the
     * own deck (library plus hand, order unused) at the smaller hand size. Slots 1/2
     * flag the two options; see docs/policy-protocol.md for the layout.
     */
    public static PolicyCandidates<Boolean> mulligan(Player player, Game game, int mulligans, PolicyClient client) {
        Map<String, Object> keep = action(client, "keep", "mulligan", null, null, game);
        Map<String, Object> mulligan = action(client, "mulligan", "mulligan", null, null, game);
        double[] k = (double[]) keep.get("features"), m = (double[]) mulligan.get("features");
        k[1] = 1;
        m[2] = 1;
        List<Card> hand = new ArrayList<>(player.getHand().getCards(game));
        HandSummary.describe(hand, hand.size(), k, game);
        List<Card> deck = new ArrayList<>(hand);
        deck.addAll(player.getLibrary().getCards(game));
        int redraw = Math.max(0, game.getMulligan().mulliganDownTo(game, player.getId()));
        HandSummary.expected(deck, redraw, m, game);
        boolean onThePlay = player.getId().equals(game.getStartingPlayerId());
        for (double[] values : Arrays.asList(k, m)) {
            values[14] = mulligans / 7.0;
            values[15] = onThePlay ? 1 : 0;
        }
        return new PolicyCandidates<Boolean>().add(false, keep).add(true, mulligan);
    }

    /** Options are the choice's keys when it has them, else its names; sorted. Throws above the candidate limit. */
    public static PolicyCandidates<String> choice(Choice choice, PolicyClient client, Game game) {
        boolean keyed = !choice.getKeyChoices().isEmpty();
        List<String> options = new ArrayList<>(keyed ? choice.getKeyChoices().keySet() : choice.getChoices());
        if (options.size() > MAX_OPTIONS && !keyed && client != null) {
            // "Name a card" lists every card in XMage. Only names in the card pool
            // can matter in these games, so offer those.
            options.retainAll(client.cardIds.keySet());
        }
        if (options.size() > MAX_OPTIONS) throw new IllegalStateException("choice has " + options.size() + " options, above the candidate limit");
        Collections.sort(options);
        PolicyCandidates<String> result = new PolicyCandidates<>();
        // Sorted option names: an alphabetical position is stable across games and
        // separates short names whose lexical bins collide (e.g. colours).
        for (int i = 0; i < options.size(); i++) {
            result.add(options.get(i), ordinal(describe(action(client, "choice-" + i, "choose", null, null, game),
                keyed ? choice.getKeyChoices().get(options.get(i)) : options.get(i)), i));
        }
        return result;
    }

    /**
     * Every value in [min, max]. Open-ended ranges ("choose a number", Sanctum
     * Prelate: 0 to 2^31) offer the smallest 256 values, a legal subset that covers
     * every mana value and life total in practice.
     */
    public static PolicyCandidates<Integer> amount(int min, int max, Ability source, Game game, PolicyClient client) {
        if ((long) max - min >= MAX_OPTIONS) max = min + 255;
        PolicyCandidates<Integer> result = new PolicyCandidates<>();
        for (int i = min; i <= max; i++) {
            Map<String, Object> candidate = action(client, "amount-" + i, "amount", source == null ? null : source.getSourceId(), null, game);
            ((double[]) candidate.get("features"))[0] = i / 64.0; // slot 0 is the amount itself, only here
            result.add(i, candidate);
        }
        return result;
    }

    /** Unselected modes whose targets can be chosen; null (stop) first once enough are selected. */
    public static PolicyCandidates<Mode> mode(Modes modes, Ability source, Game game, PolicyClient client) {
        List<Mode> options = modes.getAvailableModes(source, game).stream()
            .filter(mode -> !modes.getSelectedModes().contains(mode.getId()))
            .filter(mode -> mode.getTargets().canChoose(source.getControllerId(), source, game)).collect(Collectors.toList());
        if (modes.getMinModes() <= modes.getSelectedModes().size()) options.add(0, null);
        PolicyCandidates<Mode> result = new PolicyCandidates<>();
        // A mode's ordinal is its printed position (stop = 0), unchanged by earlier selections.
        List<Mode> printed = new ArrayList<>(modes.values());
        for (int i = 0; i < options.size(); i++) {
            Mode mode = options.get(i);
            result.add(mode, ordinal(describe(action(client, "mode-" + i, "mode", source.getSourceId(), null, game),
                mode == null ? "stop selecting modes" : mode.getEffects().getText(mode)),
                mode == null ? 0 : printed.indexOf(mode) + 1));
        }
        return result;
    }

    /** Attackers in the order their decisions are asked. */
    public static List<Permanent> attackers(Player player, Game game) {
        List<Permanent> attackers = new ArrayList<>(player.getAvailableAttackers(game));
        attackers.sort(Comparator.comparing(Permanent::getId));
        return attackers;
    }

    /** Stay (null) or one defender this attacker can attack. */
    public static PolicyCandidates<UUID> attack(Player player, Permanent attacker, Game game, PolicyClient client) {
        List<UUID> defenders = new ArrayList<>(game.getCombat().getDefenders());
        Collections.sort(defenders);
        // Stay and attack differ by their target pointer (none vs the defender).
        PolicyCandidates<UUID> result = new PolicyCandidates<UUID>().add(null, action(client, "stay", "attack", attacker.getId(), null, game));
        for (UUID defender : defenders) {
            if (player.getAvailableAttackers(defender, game).contains(attacker)) {
                result.add(defender, action(client, defender.toString(), "attack", attacker.getId(), defender, game));
            }
        }
        return result;
    }

    /** Blockers in the order their decisions are asked. */
    public static List<Permanent> blockers(Player player, Game game) {
        List<Permanent> blockers = new ArrayList<>(player.getAvailableBlockers(game));
        blockers.sort(Comparator.comparing(Permanent::getId));
        return blockers;
    }

    /** None/stop (null) or one more attacker for this blocker; empty when it can block nothing more. */
    public static PolicyCandidates<UUID> block(Permanent blocker, Set<UUID> blocked, Game game, PolicyClient client) {
        List<UUID> attackers = game.getBattlefield().getAllPermanents().stream().filter(Permanent::isAttacking)
            .filter(p -> !blocked.contains(p.getId()) && blocker.canBlock(p.getId(), game))
            .map(Permanent::getId).sorted().collect(Collectors.toList());
        PolicyCandidates<UUID> result = new PolicyCandidates<>();
        if (attackers.isEmpty()) return result;
        result.add(null, action(client, blocked.isEmpty() ? "none" : "stop", "block", blocker.getId(), null, game));
        for (UUID attacker : attackers) {
            Map<String, Object> candidate = action(client, attacker.toString(), "block", blocker.getId(), attacker, game);
            double[] values = (double[]) candidate.get("features");
            CombatGroup group = game.getCombat().findGroup(attacker);
            Permanent attacking = game.getPermanent(attacker);
            // Menace-style minimums are enforced after declaration: XMage drops
            // an AI's blockers that leave an attacker under its minimum.
            values[1] = (group == null ? 0 : group.getBlockers().size()) / 4.0;
            values[2] = (attacking == null ? 1 : attacking.getMinBlockedBy()) / 4.0;
            values[3] = blocked.size() / 4.0;
            result.add(attacker, candidate);
        }
        return result;
    }

    public static PolicyCandidates<TriggeredAbility> triggers(List<TriggeredAbility> abilities, Game game, PolicyClient client) {
        PolicyCandidates<TriggeredAbility> result = new PolicyCandidates<>();
        for (int i = 0; i < abilities.size(); i++) {
            result.add(abilities.get(i), describe(action(client, "trigger-" + i, "order", abilities.get(i).getSourceId(), null, game),
                abilities.get(i).getRule()));
        }
        return result;
    }
}
