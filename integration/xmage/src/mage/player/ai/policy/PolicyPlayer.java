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

import static mage.player.ai.policy.PolicyCandidates.describe;
import static mage.player.ai.policy.PolicyCandidates.ordinal;

/** Direct legal-action player. No MCTS node, rollout, search budget, or value search. */
public class PolicyPlayer extends ComputerPlayer implements RecordingPlayer {
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
    // Abilities the engine refused to activate (failed payment or choices) in the
    // current priority window; they are not offered again until the game moves on.
    private Set<String> rejected = new HashSet<>();
    private String rejectedWindow = "";
    // Trajectory steps of refused activations (the priority choice and any choices
    // made while XMage tried to activate). The engine rolled them back, so the
    // learner keeps them for memory and values but gives them no policy gradient.
    private List<Integer> rejectedSteps = new ArrayList<>();

    public PolicyPlayer(String name, RangeOfInfluence range, String endpoint) {
        this(name, range, endpoint, true, true);
    }

    /** Recorded batch game; sample=false plays the most likely action (evaluation, never trained on). */
    public PolicyPlayer(String name, RangeOfInfluence range, String endpoint, boolean sample) {
        this(name, range, endpoint, true, sample);
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

    protected PolicyPlayer(String name, RangeOfInfluence range, String endpoint, boolean recording, boolean sample) {
        super(name, range);
        this.endpoint = endpoint;
        this.recording = recording;
        this.sample = sample;
        this.autoTap = true; // engine legality/payment, not strategic lookahead
        this.allowMulligans = true;
    }

    protected PolicyPlayer(PolicyPlayer other) {
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
        this.rejected = other.rejected;
        this.rejectedWindow = other.rejectedWindow;
        this.rejectedSteps = other.rejectedSteps;
    }

    @Override public PolicyPlayer copy() { return new PolicyPlayer(this); }

    private static String setting(String property, String environment, String fallback) {
        String value = System.getProperty(property);
        if (value == null || value.isEmpty()) value = System.getenv(environment);
        return value == null || value.isEmpty() ? fallback : value;
    }

    @Override public void initializeKnownDeck(Game game, UUID opponent) {
        if (client == null) client = new PolicyClient(endpoint);
        encoder = new PolicyEncoder(getId(), opponent, client, game);
        encodedGame = game.getId();
        labels = new ArrayList<>();
        rejectedSteps = new ArrayList<>();
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

    private Map<String, Object> action(String id, String type, UUID source, UUID target, Game game) {
        return PolicyCandidates.action(client, id, type, source, target, game);
    }

    private int select(Game game, List<Map<String, Object>> actions, Collection<Card> visibleChoices) {
        return select(game, actions, null, visibleChoices);
    }

    private int select(Game game, List<Map<String, Object>> actions, List<?> options, Collection<Card> visibleChoices) {
        if (Thread.currentThread().isInterrupted()) throw new IllegalStateException("policy game interrupted");
        if (actions.isEmpty()) throw new IllegalStateException("empty legal-action set");
        if (game.isSimulation()) return 0; // rules-engine legality checks only
        return decide(game, actions, options, visibleChoices);
    }

    /**
     * Index of the chosen action. `options` are the engine objects behind the
     * actions when the decision has them (see PolicyCandidates), else null.
     * A subclass may take the decision elsewhere (WebPlayer: a person in a browser).
     */
    protected int decide(Game game, List<Map<String, Object>> actions, List<?> options, Collection<Card> visibleChoices) {
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

    private <T> T select(Game game, PolicyCandidates<T> candidates) {
        return candidates.options.get(select(game, candidates.actions, candidates.options, candidates.visible));
    }

    @Override public void finish(Game game, boolean terminal, boolean truncated) {
        if (!recording || encoder == null || encoder.decisions() == 0) return;
        if (terminal && truncated) throw new IllegalArgumentException("a game cannot be both terminal and truncated");
        boolean draw = game.getPlayers().values().stream().allMatch(player -> player.hasLost());
        int result = !terminal || draw ? 0 : hasWon() ? 1 : hasLost() ? -1 : 0;
        client.request("POST", "/finish", PolicyEncoder.map("game_id", encoder.gameId(),
            "player_id", getId().toString(), "result", result, "terminal", terminal, "truncated", truncated,
            "labels", terminal || truncated ? labels : Collections.emptyList(),
            "rejected_steps", terminal || truncated ? rejectedSteps : Collections.emptyList()));
    }

    @Override public boolean priority(Game game) {
        if (game.isSimulation()) { pass(game); return false; }
        game.getState().setPriorityPlayerId(getId());
        game.firePriorityEvent(getId());
        String window = game.getTurnNum() + ":" + game.getTurnStepType() + ":" + game.getStack().size()
            + ":" + game.getBattlefield().getAllPermanents().size() + ":" + getHand().size();
        if (!window.equals(rejectedWindow)) { rejected.clear(); rejectedWindow = window; }
        int firstStep = encoder == null ? 0 : encoder.decisions();
        ActivatedAbility chosen = select(game, PolicyCandidates.priority(this, game, client, rejected, tapsForMana()));
        game.setLastPriority(getId());
        if (chosen == null) { pass(game); return false; }
        if (!activateAbility(chosen, game) && !cancelledPayment()) {
            // XMage lists some abilities as playable that then fail during payment or
            // choices (the state is rolled back). Like its own AI, treat that as no
            // action, and drop the ability from this priority window's candidates.
            rejected.add(PolicyCandidates.abilityKey(chosen));
            for (int step = firstStep; step < encoder.decisions(); step++) rejectedSteps.add(step);
        }
        return true;
    }

    /**
     * True (once) when the last activation failed because the player chose not to pay
     * its mana (WebPlayer); the ability then stays on offer. The policy never cancels.
     */
    protected boolean cancelledPayment() { return false; }

    /**
     * Whether mana abilities are offered at priority, with nothing to pay (WebPlayer:
     * a person floats mana or taps for what tapping sets off). The policy taps for
     * mana only while paying.
     */
    protected boolean tapsForMana() { return false; }

    @Override protected boolean makeChoice(Outcome outcome, Target target, Ability source, Game game, Cards fromCards) {
        if (target.isChoiceCompleted(target.getAffectedAbilityControllerId(getId()), source, game, fromCards)) return target.isChosen(game);
        PolicyCandidates<UUID> candidates = PolicyCandidates.target(this, target, source, game, fromCards, client);
        if (candidates.size() == 0) return target.isChosen(game);
        UUID chosen = select(game, candidates);
        if (chosen == null) return true; // stop
        target.addTarget(chosen, source, game);
        if (!target.contains(chosen)) throw new IllegalStateException("engine rejected supplied target");
        getPlayerHistory().targetSequence.add(chosen);
        return makeChoice(outcome, target, source, game, fromCards);
    }

    @Override public boolean chooseUse(Outcome outcome, String message, String secondMessage,
            String trueText, String falseText, Ability source, Game game) {
        boolean chosen = select(game, PolicyCandidates.use(message, trueText, falseText, source, game, client));
        getPlayerHistory().useSequence.add(chosen);
        return chosen;
    }

    @Override public boolean chooseMulligan(Game game) {
        boolean chosen = select(game, PolicyCandidates.mulligan(this, game, mulligans, client));
        if (chosen) mulligans++;
        return chosen;
    }

    @Override public boolean choose(Outcome outcome, Choice choice, Game game) {
        PolicyCandidates<String> candidates = PolicyCandidates.choice(choice, client, game);
        if (candidates.size() == 0) return false;
        String selected = select(game, candidates);
        if (choice.getKeyChoices().isEmpty()) choice.setChoice(selected); else choice.setChoiceByKey(selected);
        getPlayerHistory().choiceSequence.add(selected);
        return true;
    }

    @Override protected int makeChoiceAmount(int min, int max, Game game, Ability source, boolean isManaPay) {
        if (min >= max) return min;
        int chosen = select(game, PolicyCandidates.amount(min, max, source, game, client));
        getPlayerHistory().numSequence.add(chosen - min);
        return chosen;
    }

    @Override public Mode chooseMode(Modes modes, Ability source, Game game) {
        PolicyCandidates<Mode> candidates = PolicyCandidates.mode(modes, source, game, client);
        return candidates.size() == 0 ? null : select(game, candidates);
    }

    @Override public void selectAttackers(Game game, UUID attackingPlayerId) {
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_ATTACKERS_STEP_PRE, null, null, attackingPlayerId));
        if (game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_ATTACKERS, attackingPlayerId, attackingPlayerId))) return;
        for (Permanent attacker : PolicyCandidates.attackers(this, game)) {
            UUID defender = select(game, PolicyCandidates.attack(this, attacker, game, client));
            if (defender != null) declareAttacker(attacker.getId(), defender, game, false);
        }
        game.getPlayers().resetPassed();
    }

    @Override public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_BLOCKERS_STEP_PRE, null, null, defendingPlayerId));
        if (game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_BLOCKERS, defendingPlayerId, defendingPlayerId))) return;
        for (Permanent blocker : PolicyCandidates.blockers(this, game)) {
            // Sequential per blocker: a creature that can block additional attackers
            // (max blocks > 1, or 0 for any number) chooses again until it stops.
            int limit = blocker.getMaxBlocks();
            Set<UUID> blocked = new HashSet<>();
            while (limit == 0 || blocked.size() < limit) {
                PolicyCandidates<UUID> candidates = PolicyCandidates.block(blocker, blocked, game, client);
                if (candidates.size() == 0) break;
                UUID attacker = select(game, candidates);
                if (attacker == null) break;
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
        return select(game, PolicyCandidates.triggers(abilities, game, client));
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
        Map<String, Object> candidate = ordinal(action(id, "choose", top == null ? null : top.getId(), null, game), index);
        double[] values = (double[]) candidate.get("features");
        int lands = 0, creatures = 0, manaValue = 0, power = 0;
        for (Card card : cards) {
            if (card.isLand(game)) lands++;
            if (card.isCreature(game)) { creatures++; power += Math.max(0, card.getPower().getValue()); }
            manaValue += card.getManaValue();
        }
        values[3] = cards.size() / 60.0;
        values[4] = lands / 8.0;
        values[5] = (cards.size() - lands) / 8.0;
        values[6] = creatures / 8.0;
        values[7] = manaValue / 32.0;
        values[8] = (top == null ? 0 : top.getManaValue()) / 16.0;
        values[9] = power / 20.0;
        return candidate;
    }

    @Override public SpellAbility chooseAbilityForCast(Card card, Game game, boolean noMana) {
        List<SpellAbility> usable = new ArrayList<>(PlayerImpl.getCastableSpellAbilities(
            game, getId(), card, game.getState().getZone(card.getId()), noMana).values());
        if (usable.isEmpty()) return null;
        usable.sort(Comparator.comparing(Ability::toString));
        List<Map<String, Object>> actions = new ArrayList<>();
        for (int i = 0; i < usable.size(); i++) actions.add(ordinal(describe(action("cast-variant-" + i, "cast", card.getId(), null, game), usable.get(i).getRule()), i));
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
            if (target.isChosen(game)) actions.add(action("stop", "target", source.getSourceId(), null, game));
            for (UUID id : candidates) actions.add(action(id.toString(), "target", source.getSourceId(), id, game));
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
        // XMage reads the result as an index into the map's iteration order; the keys
        // are object/effect identifiers, not numbers.
        List<String> keys = new ArrayList<>(effects.keySet());
        if (keys.size() <= 1) return 0;
        List<String> sorted = new ArrayList<>(keys);
        Collections.sort(sorted);
        List<Map<String, Object>> actions = new ArrayList<>();
        for (int i = 0; i < sorted.size(); i++) actions.add(describe(action("replacement-" + i, "order", null, null, game), effects.get(sorted.get(i))));
        return keys.indexOf(sorted.get(select(game, actions, null)));
    }
}
