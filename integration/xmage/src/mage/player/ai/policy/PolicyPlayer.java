package mage.player.ai.policy;

import mage.MageObject;
import mage.abilities.*;
import mage.abilities.common.PassAbility;
import mage.cards.*;
import mage.choices.Choice;
import mage.constants.*;
import mage.game.Game;
import mage.game.combat.CombatGroup;
import mage.game.events.GameEvent;
import mage.game.permanent.Permanent;
import mage.player.ai.ComputerPlayer;
import mage.target.Target;
import mage.target.TargetAmount;
import mage.players.PlayerImpl;
import java.util.*;
import java.util.stream.Collectors;

/** Direct legal-action player. No MCTS node, rollout, search budget, or value search. */
public final class PolicyPlayer extends ComputerPlayer {
    private final String endpoint;
    // Self-play records privileged labels and reports the result; server games
    // against people are inference only.
    private final boolean recording;
    private final boolean sample;
    private transient PolicyClient client;
    private transient PolicyEncoder encoder;
    private transient UUID encodedGame;
    private List<Map<String, Object>> labels = new ArrayList<>();
    private int mulligans = 0;

    public PolicyPlayer(String name, RangeOfInfluence range, String endpoint) {
        this(name, range, endpoint, true, true);
    }

    /**
     * XMage server constructor (config.xml playerType). The endpoint comes from
     * -Dmagezero.policy.endpoint or MAGEZERO_POLICY_ENDPOINT; -Dmagezero.policy.sample=false
     * plays the most likely action instead of sampling.
     */
    public PolicyPlayer(String name, RangeOfInfluence range, int skill) {
        this(name, range, setting("magezero.policy.endpoint", "MAGEZERO_POLICY_ENDPOINT", "http://127.0.0.1:50052"),
             false, !"false".equalsIgnoreCase(setting("magezero.policy.sample", "MAGEZERO_POLICY_SAMPLE", "true")));
    }

    private PolicyPlayer(String name, RangeOfInfluence range, String endpoint, boolean recording, boolean sample) {
        super(name, range);
        this.endpoint = endpoint;
        this.recording = recording;
        this.sample = sample;
        this.autoTap = true; // engine legality/payment, not strategic lookahead
        this.allowMulligans = true;
    }

    private PolicyPlayer(PolicyPlayer other) {
        super(other);
        this.endpoint = other.endpoint;
        this.recording = other.recording;
        this.sample = other.sample;
        // Shared, not copied: XMage replaces live players with saved copies when it
        // restores a state (failed payments, human undo), and the restored player must
        // keep the same recurrent session. Simulation copies never call the policy.
        this.client = other.client;
        this.encoder = other.encoder;
        this.encodedGame = other.encodedGame;
        this.labels = other.labels;
        this.mulligans = other.mulligans;
    }

    @Override public PolicyPlayer copy() { return new PolicyPlayer(this); }

    private static String setting(String property, String environment, String fallback) {
        String value = System.getProperty(property);
        if (value == null || value.isEmpty()) value = System.getenv(environment);
        return value == null || value.isEmpty() ? fallback : value;
    }

    public void initializeKnownDeck(Game game, UUID opponent) {
        if (client == null) client = new PolicyClient(endpoint);
        encoder = new PolicyEncoder(getId(), opponent, client, game);
        encodedGame = game.getId();
        labels = new ArrayList<>();
        mulligans = 0;
    }

    /**
     * Server games never call initializeKnownDeck. GameImpl.init calls beginTurn for
     * every player before shuffling and opening hands, so the announced deck lists are
     * read at the same point as in self-play. A new game of a match re-initializes.
     */
    @Override public void beginTurn(Game game) {
        super.beginTurn(game);
        if (game.isSimulation() || game.getId().equals(encodedGame)) return;
        List<UUID> opponents = new ArrayList<>(game.getOpponents(getId()));
        if (opponents.size() != 1) throw new IllegalStateException("policy player supports two-player games only");
        initializeKnownDeck(game, opponents.get(0));
    }

    private Map<String, Object> action(String id, String type, UUID source, UUID target, Game game, double argument) {
        double[] features = new double[16];
        features[0] = argument / 64.0;
        MageObject object = source == null ? null : game.getObject(source);
        boolean concealed = object instanceof Card && ((Card) object).isFaceDown(game);
        int cardId = object == null || client == null || concealed ? 0 : client.cardIds.getOrDefault(object.getName(), 0);
        return PolicyEncoder.map("id", id, "type", type,
            "source", source == null ? null : source.toString(), "target", target == null ? null : target.toString(),
            "card_id", cardId, "features", features);
    }

    private Map<String, Object> describe(Map<String, Object> action, String text) {
        // Candidate semantics supplement card/pointer embeddings. Stable lexical
        // bins distinguish modes and named choices across changing candidate sets.
        double[] values = (double[]) action.get("features");
        for (String word : String.valueOf(text).toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
            if (!word.isEmpty()) values[3 + Math.floorMod(word.hashCode(), 13)] += 1;
        }
        for (int i = 3; i < 16; i++) values[i] = Math.tanh(values[i] / 4.0);
        return action;
    }

    private int select(Game game, List<Map<String, Object>> actions, Collection<Card> visibleChoices) {
        if (Thread.currentThread().isInterrupted()) throw new IllegalStateException("policy game interrupted");
        if (actions.isEmpty()) throw new IllegalStateException("empty legal-action set");
        if (game.isSimulation()) return 0; // rules-engine legality checks only
        if (encoder == null) throw new IllegalStateException("policy player not initialized before game.start");
        if (actions.size() == 1) {
            // Forced: no inference, label, or trajectory step (the observation step
            // counter is untouched). History still records the choice, and the
            // policy receives it with its next real observation.
            encoder.recordChoice(actions.get(0), game);
            return 0;
        }
        Map<String, Object> observation = encoder.observation(game, actions, visibleChoices);
        String chosen = client.request("POST", "/act", PolicyEncoder.map("observation", observation,
                                   "sample", sample)).get("action_id").getAsString();
        if (recording) labels.add(encoder.trainingLabel(game));
        for (int i = 0; i < actions.size(); i++) if (actions.get(i).get("id").equals(chosen)) {
            encoder.recordChoice(actions.get(i), game);
            return i;
        }
        throw new IllegalStateException("policy returned an action outside the supplied legal set");
    }

    /**
     * terminal: the game reached a real result. truncated: an error-free game
     * stopped by the turn limit; it is recorded with result 0 and bootstrapped by
     * the learner, never scored as a draw. Neither: aborted, trajectory discarded.
     */
    public void finish(Game game, boolean terminal, boolean truncated) {
        if (!recording || encoder == null || encoder.decisions() == 0) return;
        if (terminal && truncated) throw new IllegalArgumentException("a game cannot be both terminal and truncated");
        boolean draw = game.getPlayers().values().stream().allMatch(player -> player.hasLost());
        int result = !terminal || draw ? 0 : hasWon() ? 1 : hasLost() ? -1 : 0;
        client.request("POST", "/finish", PolicyEncoder.map("game_id", encoder.gameId(),
            "player_id", getId().toString(), "result", result, "terminal", terminal, "truncated", truncated,
            "labels", terminal || truncated ? labels : Collections.emptyList()));
    }

    @Override public boolean priority(Game game) {
        if (game.isSimulation()) { pass(game); return false; }
        game.getState().setPriorityPlayerId(getId());
        game.firePriorityEvent(getId());
        List<ActivatedAbility> abilities = new ArrayList<>(getPlayable(game, true));
        abilities.removeIf(Ability::isManaAbility);
        abilities.sort(Comparator.comparing((Ability a) -> String.valueOf(a.getSourceId())).thenComparing(Ability::toString));
        abilities.add(0, new PassAbility());
        List<Map<String, Object>> choices = new ArrayList<>();
        for (int i = 0; i < abilities.size(); i++) {
            ActivatedAbility ability = abilities.get(i);
            String type = ability instanceof PassAbility ? "pass" : ability instanceof SpellAbility ? "cast" : "activate";
            Map<String, Object> candidate = action("priority-" + i, type, ability.getSourceId(), null, game, i);
            ((double[]) candidate.get("features"))[1] = ability.getManaCosts().manaValue() / 16.0;
            choices.add(describe(candidate, ability.getRule()));
        }
        ActivatedAbility chosen = abilities.get(select(game, choices, null));
        game.setLastPriority(getId());
        if (chosen instanceof PassAbility) { pass(game); return false; }
        if (!activateAbility(chosen, game)) throw new IllegalStateException("engine rejected a supplied priority action");
        return true;
    }

    @Override protected boolean makeChoice(Outcome outcome, Target target, Ability source, Game game, Cards fromCards) {
        if (target.isChoiceCompleted(target.getAffectedAbilityControllerId(getId()), source, game, fromCards)) return target.isChosen(game);
        List<UUID> candidates = target.possibleTargets(target.getAffectedAbilityControllerId(getId()), source, game, fromCards)
                .stream().filter(id -> !target.contains(id)).sorted().collect(Collectors.toList());
        if (candidates.isEmpty()) return target.isChosen(game);
        List<Map<String, Object>> choices = new ArrayList<>();
        // The target prompt (e.g. "put on the bottom of your library") gives the
        // choice its purpose; the pointer gives the object.
        String prompt = target.getMessage(game);
        if (target.isChosen(game)) choices.add(describe(action("stop", "target", source == null ? null : source.getSourceId(), null, game, 0), prompt));
        for (int i = 0; i < candidates.size(); i++) choices.add(describe(action("target-" + i, "target",
            source == null ? null : source.getSourceId(), candidates.get(i), game, i + 1), prompt));
        int selected = select(game, choices, fromCards == null ? null : fromCards.getCards(game));
        if (target.isChosen(game)) { if (selected == 0) return true; selected--; }
        UUID chosen = candidates.get(selected);
        target.addTarget(chosen, source, game);
        if (!target.contains(chosen)) throw new IllegalStateException("engine rejected supplied target");
        getPlayerHistory().targetSequence.add(chosen);
        return makeChoice(outcome, target, source, game, fromCards);
    }

    @Override public boolean chooseUse(Outcome outcome, String message, String secondMessage,
            String trueText, String falseText, Ability source, Game game) {
        UUID reference = source == null ? null : source.getSourceId();
        int chosen = select(game, Arrays.asList(describe(action("no", "use", reference, null, game, 0), message + " " + falseText),
                                               describe(action("yes", "use", reference, null, game, 1), message + " " + trueText)), null);
        getPlayerHistory().useSequence.add(chosen == 1);
        return chosen == 1;
    }

    /**
     * Keep describes the current hand; mulligan describes an average redraw of the
     * own deck (library plus hand, order unused) at the smaller hand size. Slots 1/2
     * flag the two options; see docs/policy-protocol.md for the layout.
     */
    @Override public boolean chooseMulligan(Game game) {
        Map<String, Object> keep = action("keep", "mulligan", null, null, game, 0);
        Map<String, Object> mulligan = action("mulligan", "mulligan", null, null, game, 1);
        double[] k = (double[]) keep.get("features"), m = (double[]) mulligan.get("features");
        k[1] = 1;
        m[2] = 1;
        List<Card> hand = new ArrayList<>(getHand().getCards(game));
        HandSummary.describe(hand, hand.size(), k, game);
        List<Card> deck = new ArrayList<>(hand);
        deck.addAll(getLibrary().getCards(game));
        int redraw = Math.max(0, game.getMulligan().mulliganDownTo(game, getId()));
        HandSummary.expected(deck, redraw, m, game);
        boolean onThePlay = getId().equals(game.getStartingPlayerId());
        for (double[] values : Arrays.asList(k, m)) {
            values[14] = mulligans / 7.0;
            values[15] = onThePlay ? 1 : 0;
        }
        boolean chosen = select(game, Arrays.asList(keep, mulligan), null) == 1;
        if (chosen) mulligans++;
        return chosen;
    }

    @Override public boolean choose(Outcome outcome, Choice choice, Game game) {
        List<String> options = new ArrayList<>(choice.getKeyChoices().isEmpty() ? choice.getChoices() : choice.getKeyChoices().keySet());
        Collections.sort(options);
        if (options.isEmpty()) return false;
        List<Map<String, Object>> actions = new ArrayList<>();
        for (int i = 0; i < options.size(); i++) actions.add(describe(action("choice-" + i, "choose", null, null, game, i),
            choice.getKeyChoices().isEmpty() ? options.get(i) : choice.getKeyChoices().get(options.get(i))));
        String selected = options.get(select(game, actions, null));
        if (choice.getKeyChoices().isEmpty()) choice.setChoice(selected); else choice.setChoiceByKey(selected);
        getPlayerHistory().choiceSequence.add(selected);
        return true;
    }

    @Override protected int makeChoiceAmount(int min, int max, Game game, Ability source, boolean isManaPay) {
        if (min >= max) return min;
        if ((long) max - min > 2047) throw new IllegalStateException("amount choice exceeds candidate limit; use a hierarchical amount decoder");
        List<Map<String, Object>> actions = new ArrayList<>();
        for (int i = min; i <= max; i++) actions.add(action("amount-" + i, "amount",
            source == null ? null : source.getSourceId(), null, game, i));
        int chosen = min + select(game, actions, null);
        getPlayerHistory().numSequence.add(chosen - min);
        return chosen;
    }

    @Override public Mode chooseMode(Modes modes, Ability source, Game game) {
        List<Mode> options = modes.getAvailableModes(source, game).stream()
            .filter(mode -> !modes.getSelectedModes().contains(mode.getId()))
            .filter(mode -> mode.getTargets().canChoose(source.getControllerId(), source, game)).collect(Collectors.toList());
        if (modes.getMinModes() <= modes.getSelectedModes().size()) options.add(0, null);
        if (options.isEmpty()) return null;
        List<Map<String, Object>> actions = new ArrayList<>();
        for (int i = 0; i < options.size(); i++) actions.add(describe(action("mode-" + i, "mode", source.getSourceId(), null, game, i),
            options.get(i) == null ? "stop selecting modes" : options.get(i).getEffects().getText(options.get(i))));
        return options.get(select(game, actions, null));
    }

    @Override public void selectAttackers(Game game, UUID attackingPlayerId) {
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_ATTACKERS_STEP_PRE, null, null, attackingPlayerId));
        if (game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_ATTACKERS, attackingPlayerId, attackingPlayerId))) return;
        List<Permanent> attackers = new ArrayList<>(getAvailableAttackers(game));
        attackers.sort(Comparator.comparing(Permanent::getId));
        for (Permanent attacker : attackers) {
            List<UUID> defenders = new ArrayList<>(game.getCombat().getDefenders());
            Collections.sort(defenders);
            List<Map<String, Object>> actions = new ArrayList<>();
            actions.add(action("stay", "attack", attacker.getId(), null, game, 0));
            for (UUID defender : defenders) {
                if (getAvailableAttackers(defender, game).contains(attacker)) {
                    actions.add(action(defender.toString(), "attack", attacker.getId(), defender, game, 1));
                }
            }
            int chosen = select(game, actions, null);
            if (chosen > 0) declareAttacker(attacker.getId(), UUID.fromString((String) actions.get(chosen).get("id")), game, false);
        }
        game.getPlayers().resetPassed();
    }

    @Override public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_BLOCKERS_STEP_PRE, null, null, defendingPlayerId));
        if (game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_BLOCKERS, defendingPlayerId, defendingPlayerId))) return;
        List<Permanent> blockers = new ArrayList<>(getAvailableBlockers(game));
        blockers.sort(Comparator.comparing(Permanent::getId));
        for (Permanent blocker : blockers) {
            // Sequential per blocker: a creature that can block additional attackers
            // (max blocks > 1, or 0 for any number) chooses again until it stops.
            int limit = blocker.getMaxBlocks();
            Set<UUID> blocked = new HashSet<>();
            while (limit == 0 || blocked.size() < limit) {
                List<UUID> attackers = game.getBattlefield().getAllPermanents().stream().filter(Permanent::isAttacking)
                    .filter(p -> !blocked.contains(p.getId()) && blocker.canBlock(p.getId(), game))
                    .map(Permanent::getId).sorted().collect(Collectors.toList());
                if (attackers.isEmpty()) break;
                List<Map<String, Object>> actions = new ArrayList<>();
                actions.add(action(blocked.isEmpty() ? "none" : "stop", "block", blocker.getId(), null, game, 0));
                for (UUID attacker : attackers) {
                    Map<String, Object> candidate = action(attacker.toString(), "block", blocker.getId(), attacker, game, 1);
                    double[] values = (double[]) candidate.get("features");
                    CombatGroup group = game.getCombat().findGroup(attacker);
                    Permanent attacking = game.getPermanent(attacker);
                    // Menace-style minimums are enforced after declaration: XMage drops
                    // an AI's blockers that leave an attacker under its minimum.
                    values[1] = (group == null ? 0 : group.getBlockers().size()) / 4.0;
                    values[2] = (attacking == null ? 1 : attacking.getMinBlockedBy()) / 4.0;
                    values[3] = blocked.size() / 4.0;
                    actions.add(candidate);
                }
                int chosen = select(game, actions, null);
                if (chosen == 0) break;
                UUID attacker = UUID.fromString((String) actions.get(chosen).get("id"));
                declareBlocker(defendingPlayerId, blocker.getId(), attacker, game);
                CombatGroup group = game.getCombat().findGroup(attacker);
                if (group == null || !group.getBlockers().contains(blocker.getId())) break; // engine refused
                blocked.add(attacker);
            }
        }
        game.getPlayers().resetPassed();
    }

    @Override public TriggeredAbility chooseTriggeredAbility(List<TriggeredAbility> abilities, Game game) {
        if (abilities.isEmpty()) return null;
        List<Map<String, Object>> actions = new ArrayList<>();
        for (int i = 0; i < abilities.size(); i++) actions.add(describe(action("trigger-" + i, "order", abilities.get(i).getSourceId(), null, game, i), abilities.get(i).getRule()));
        return abilities.get(select(game, actions, null));
    }

    @Override public boolean choosePile(Outcome outcome, String message, List<? extends Card> first,
            List<? extends Card> second, Game game) {
        List<Card> visible = new ArrayList<>();
        visible.addAll(first); visible.addAll(second);
        return select(game, Arrays.asList(pile("first", first, game, 0), pile("second", second, game, 1)), visible) == 0;
    }

    /**
     * Pile contents as aggregates (slots 2-8); the source pointer and card ID name its
     * highest mana-value card, so equal-sized piles are distinguishable.
     */
    private Map<String, Object> pile(String id, List<? extends Card> cards, Game game, int index) {
        Card top = cards.stream().max(Comparator.comparingInt(Card::getManaValue).thenComparing(Card::getId)).orElse(null);
        Map<String, Object> candidate = action(id, "choose", top == null ? null : top.getId(), null, game, index);
        double[] values = (double[]) candidate.get("features");
        int lands = 0, creatures = 0, manaValue = 0, power = 0;
        for (Card card : cards) {
            if (card.isLand(game)) lands++;
            if (card.isCreature(game)) { creatures++; power += Math.max(0, card.getPower().getValue()); }
            manaValue += card.getManaValue();
        }
        values[2] = cards.size() / 60.0;
        values[3] = lands / 8.0;
        values[4] = (cards.size() - lands) / 8.0;
        values[5] = creatures / 8.0;
        values[6] = manaValue / 32.0;
        values[7] = (top == null ? 0 : top.getManaValue()) / 16.0;
        values[8] = power / 20.0;
        return candidate;
    }

    @Override public SpellAbility chooseAbilityForCast(Card card, Game game, boolean noMana) {
        List<SpellAbility> usable = new ArrayList<>(PlayerImpl.getCastableSpellAbilities(
            game, getId(), card, game.getState().getZone(card.getId()), noMana).values());
        if (usable.isEmpty()) return null;
        usable.sort(Comparator.comparing(Ability::toString));
        List<Map<String, Object>> actions = new ArrayList<>();
        for (int i = 0; i < usable.size(); i++) actions.add(describe(action("cast-variant-" + i, "cast", card.getId(), null, game, i), usable.get(i).getRule()));
        return usable.get(select(game, actions, Collections.singletonList(card)));
    }

    @Override public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
        target.prepareAmount(source, game);
        if (target.getAmountRemaining() <= 0 || target.getMaxNumberOfTargets() == 0) return false;
        UUID controller = target.getAffectedAbilityControllerId(getId());
        while (!target.isChoiceCompleted(controller, source, game, null)) {
            List<UUID> candidates = target.possibleTargets(controller, source, game, null).stream()
                .filter(id -> !target.contains(id)).sorted().collect(Collectors.toList());
            if (candidates.isEmpty()) return target.isChosen(game);
            List<Map<String, Object>> actions = new ArrayList<>();
            if (target.isChosen(game)) actions.add(action("stop", "target", source.getSourceId(), null, game, 0));
            for (UUID id : candidates) actions.add(action(id.toString(), "target", source.getSourceId(), id, game, 1));
            Map<String, Object> chosen = actions.get(select(game, actions, null));
            if ("stop".equals(chosen.get("id"))) return true;
            int remaining = target.getAmountRemaining();
            // Reserve one unit for every still-mandatory target, and allocate all
            // remaining units when the final available target is reached.
            int mandatory = Math.max(0, target.getMinNumberOfTargets() - target.getSize() - 1);
            int maximum = remaining - mandatory;
            if (maximum < 1) throw new IllegalStateException("infeasible target amount allocation");
            int minimum = (candidates.size() == 1 || target.getSize() + 1 == target.getMaxNumberOfTargets()) ? remaining : 1;
            int amount = makeChoiceAmount(minimum, maximum, game, source, false);
            target.addTarget(UUID.fromString((String) chosen.get("id")), amount, source, game);
        }
        return target.isChosen(game);
    }

    @Override public int chooseReplacementEffect(Map<String, String> effects, Map<String, MageObject> objects, Game game) {
        List<String> keys = new ArrayList<>(effects.keySet());
        Collections.sort(keys);
        List<Map<String, Object>> actions = new ArrayList<>();
        for (int i = 0; i < keys.size(); i++) actions.add(describe(action("replacement-" + i, "order", null, null, game, i), effects.get(keys.get(i))));
        return Integer.parseInt(keys.get(select(game, actions, null)));
    }
}
