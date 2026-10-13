package mage.player.ai;

import mage.abilities.*;
import mage.abilities.common.PassAbility;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.Outcome;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.combat.CombatGroup;
import mage.game.permanent.Permanent;
import mage.player.ai.policy.*;
import mage.target.Target;
import java.util.*;

/**
 * XMage's Mad AI (ComputerPlayer7) playing normally while its decisions are recorded
 * as labels over the policy's own candidate lists (PolicyCandidates), for behaviour
 * cloning. Each recorded decision is POSTed to the policy service's /record with the
 * observation the policy would have seen; /finish writes the trajectory.
 *
 * The AI decides first, then its decision is mapped to a candidate index:
 * - priority: each ability its search activates, and its pass;
 * - combat: its attack/block plan is captured, then replayed one creature at a time
 *   in the policy's order (as the policy declares them);
 * - targets: chosen on a copy of the target, then added one at a time (with stop);
 * - use, mulligan, named choice, amount, mode, trigger order: the returned answer.
 * Decisions the policy is never asked for, or answers outside the candidate list
 * (e.g. targets the AI's search fixed in advance), are played but not recorded.
 */
public final class TeacherPlayer extends ComputerPlayer7 implements RecordingPlayer {

    /** Shared by every copy: XMage restores saved player copies during rules processing. */
    private static final class Session {
        final String endpoint;
        PolicyClient client;
        PolicyEncoder encoder;
        List<Map<String, Object>> labels = new ArrayList<>();
        int mulligans;
        int records;
        int priorityDepth;   // inside the AI's priority()
        int activating;      // inside an activation it already recorded
        Map<UUID, UUID> attackPlan;          // attacker -> defender, while capturing
        Map<UUID, List<UUID>> blockPlan;     // blocker -> attackers, while capturing
        Session(String endpoint) { this.endpoint = endpoint; }
    }

    private final Session session;

    public TeacherPlayer(String name, RangeOfInfluence range, int skill, String endpoint) {
        super(name, range, skill);
        this.session = new Session(endpoint);
    }

    private TeacherPlayer(TeacherPlayer other) {
        super(other);
        this.session = other.session;
    }

    @Override public TeacherPlayer copy() { return new TeacherPlayer(this); }

    @Override public void initializeKnownDeck(Game game, UUID opponent) {
        if (session.client == null) session.client = new PolicyClient(session.endpoint);
        session.encoder = new PolicyEncoder(getId(), opponent, session.client, game);
        session.labels = new ArrayList<>();
        session.mulligans = 0;
        session.records = 0;
    }

    @Override public void finish(Game game, boolean terminal, boolean truncated) {
        if (session.encoder == null || session.records == 0) return;
        if (terminal && truncated) throw new IllegalArgumentException("a game cannot be both terminal and truncated");
        boolean draw = game.getPlayers().values().stream().allMatch(player -> player.hasLost());
        int result = !terminal || draw ? 0 : hasWon() ? 1 : hasLost() ? -1 : 0;
        session.client.request("POST", "/finish", PolicyEncoder.map("game_id", session.encoder.gameId(),
            "player_id", getId().toString(), "result", result, "terminal", terminal, "truncated", truncated,
            "labels", terminal || truncated ? session.labels : Collections.emptyList()));
    }

    private boolean live(Game game) {
        return !game.isSimulation() && session.encoder != null;
    }

    /** Records the AI's choice `label` among `candidates`, exactly as PolicyPlayer.select would see it. */
    private void record(Game game, PolicyCandidates<?> candidates, int label) {
        if (Thread.currentThread().isInterrupted()) throw new IllegalStateException("teacher game interrupted");
        if (label < 0 || label >= candidates.size()) return;
        if (candidates.size() == 1) {
            session.encoder.recordChoice(candidates.actions.get(0), game);
            return;
        }
        Map<String, Object> observation = session.encoder.observation(game, candidates.actions, candidates.visible);
        session.client.request("POST", "/record", PolicyEncoder.map("observation", observation,
            "action_id", candidates.actions.get(label).get("id")));
        session.labels.add(session.encoder.trainingLabel(game));
        session.records++;
        session.encoder.recordChoice(candidates.actions.get(label), game);
    }

    // ---- priority: the abilities the AI's search activates, then its pass ----

    @Override public boolean priority(Game game) {
        session.priorityDepth++;
        try {
            return super.priority(game);
        } finally {
            session.priorityDepth--;
        }
    }

    private boolean recordsPriority(Game game) {
        return live(game) && session.priorityDepth > 0 && session.activating == 0;
    }

    @Override public boolean activateAbility(ActivatedAbility ability, Game game) {
        if (!recordsPriority(game) || ability.isManaAbility()) return super.activateAbility(ability, game);
        PolicyCandidates<ActivatedAbility> candidates = PolicyCandidates.priority(this, game, session.client, Collections.emptySet());
        int label = 0; // PassAbility
        if (!(ability instanceof PassAbility)) {
            String key = PolicyCandidates.abilityKey(ability);
            label = -1;
            for (int i = 1; i < candidates.size(); i++) {
                if (PolicyCandidates.abilityKey(candidates.options.get(i)).equals(key)) { label = i; break; }
            }
        }
        record(game, candidates, label);
        // The activation's own pass (PassAbility) and mana payments are part of this decision.
        session.activating++;
        try {
            return super.activateAbility(ability, game);
        } finally {
            session.activating--;
        }
    }

    @Override public void pass(Game game) {
        if (recordsPriority(game)) record(game, PolicyCandidates.priority(this, game, session.client, Collections.emptySet()), 0);
        super.pass(game);
    }

    // ---- combat: capture the AI's plan, then declare it the way the policy does ----

    @Override public void declareAttacker(UUID attackerId, UUID defenderId, Game game, boolean allowUndo) {
        if (live(game) && session.attackPlan != null) {
            // The AI skips creatures already attacking; the first assignment wins.
            session.attackPlan.putIfAbsent(attackerId, defenderId);
            return;
        }
        super.declareAttacker(attackerId, defenderId, game, allowUndo);
    }

    @Override public void selectAttackers(Game game, UUID attackingPlayerId) {
        if (!live(game)) { super.selectAttackers(game, attackingPlayerId); return; }
        Map<UUID, UUID> plan = new LinkedHashMap<>();
        session.attackPlan = plan;
        try {
            super.selectAttackers(game, attackingPlayerId);
        } finally {
            session.attackPlan = null;
        }
        for (Permanent attacker : PolicyCandidates.attackers(this, game)) {
            PolicyCandidates<UUID> candidates = PolicyCandidates.attack(this, attacker, game, session.client);
            int label = Math.max(0, candidates.options.indexOf(plan.get(attacker.getId())));
            record(game, candidates, label);
            if (label > 0) super.declareAttacker(attacker.getId(), candidates.options.get(label), game, false);
        }
        game.getPlayers().resetPassed();
    }

    @Override public void declareBlocker(UUID defenderId, UUID blockerId, UUID attackerId, Game game, boolean allowUndo) {
        if (live(game) && session.blockPlan != null) {
            session.blockPlan.computeIfAbsent(blockerId, id -> new ArrayList<>()).add(attackerId);
            return;
        }
        super.declareBlocker(defenderId, blockerId, attackerId, game, allowUndo);
    }

    @Override public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        if (!live(game)) { super.selectBlockers(source, game, defendingPlayerId); return; }
        Map<UUID, List<UUID>> plan = new LinkedHashMap<>();
        session.blockPlan = plan;
        try {
            super.selectBlockers(source, game, defendingPlayerId);
        } finally {
            session.blockPlan = null;
        }
        for (Permanent blocker : PolicyCandidates.blockers(this, game)) {
            List<UUID> wanted = new ArrayList<>(plan.getOrDefault(blocker.getId(), Collections.emptyList()));
            int limit = blocker.getMaxBlocks();
            Set<UUID> blocked = new HashSet<>();
            while (limit == 0 || blocked.size() < limit) {
                PolicyCandidates<UUID> candidates = PolicyCandidates.block(blocker, blocked, game, session.client);
                if (candidates.size() == 0) break;
                UUID next = wanted.stream().filter(candidates.options::contains).findFirst().orElse(null);
                record(game, candidates, Math.max(0, candidates.options.indexOf(next)));
                if (next == null) break;
                super.declareBlocker(defendingPlayerId, blocker.getId(), next, game, false);
                CombatGroup group = game.getCombat().findGroup(next);
                if (group == null || !group.getBlockers().contains(blocker.getId())) break; // engine refused
                blocked.add(next);
                wanted.remove(next);
            }
        }
        game.getPlayers().resetPassed();
    }

    // ---- targets: the AI chooses on a copy; the targets are then added one at a time ----

    @Override protected boolean makeChoice(Outcome outcome, Target target, Ability source, Game game, Cards fromCards) {
        if (!live(game)) return super.makeChoice(outcome, target, source, game, fromCards);
        UUID controller = target.getAffectedAbilityControllerId(getId());
        if (target.isChoiceCompleted(controller, source, game, fromCards)) return target.isChosen(game);
        Target plan = target.copy();
        super.makeChoice(outcome, plan, source, game, fromCards);
        List<UUID> wanted = new ArrayList<>(plan.getTargets());
        wanted.removeAll(target.getTargets());
        while (!target.isChoiceCompleted(controller, source, game, fromCards)) {
            PolicyCandidates<UUID> candidates = PolicyCandidates.target(this, target, source, game, fromCards, session.client);
            if (candidates.size() == 0) break;
            UUID next = wanted.stream().filter(candidates.options::contains).findFirst().orElse(null);
            int label = candidates.options.indexOf(next); // null: stop, when offered
            if (label < 0) break;
            record(game, candidates, label);
            if (next == null) return true;
            target.addTarget(next, source, game);
            if (!target.contains(next)) break;
            wanted.remove(next);
        }
        return target.isChosen(game);
    }

    // ---- single answers: the AI's return value is the label ----

    @Override public boolean chooseUse(Outcome outcome, String message, String secondMessage,
                                       String trueText, String falseText, Ability source, Game game) {
        boolean chosen = super.chooseUse(outcome, message, secondMessage, trueText, falseText, source, game);
        if (live(game)) record(game, PolicyCandidates.use(message, trueText, falseText, source, game, session.client), chosen ? 1 : 0);
        return chosen;
    }

    @Override public boolean chooseMulligan(Game game) {
        boolean chosen = super.chooseMulligan(game);
        if (live(game)) {
            record(game, PolicyCandidates.mulligan(this, game, session.mulligans, session.client), chosen ? 1 : 0);
            if (chosen) session.mulligans++;
        }
        return chosen;
    }

    @Override public boolean choose(Outcome outcome, Choice choice, Game game) {
        if (!live(game)) return super.choose(outcome, choice, game);
        PolicyCandidates<String> candidates;
        try {
            candidates = PolicyCandidates.choice(choice, session.client, game);
        } catch (IllegalStateException tooMany) {
            return super.choose(outcome, choice, game);
        }
        boolean result = super.choose(outcome, choice, game);
        if (result) {
            String selected = choice.getKeyChoices().isEmpty() ? choice.getChoice() : choice.getChoiceKey();
            record(game, candidates, candidates.options.indexOf(selected));
        }
        return result;
    }

    @Override protected int makeChoiceAmount(int min, int max, Game game, Ability source, boolean isManaPay) {
        int chosen = super.makeChoiceAmount(min, max, game, source, isManaPay);
        if (live(game) && min < max) {
            PolicyCandidates<Integer> candidates = PolicyCandidates.amount(min, max, source, game, session.client);
            record(game, candidates, candidates.options.indexOf(chosen));
        }
        return chosen;
    }

    @Override public Mode chooseMode(Modes modes, Ability source, Game game) {
        if (!live(game)) return super.chooseMode(modes, source, game);
        PolicyCandidates<Mode> candidates = PolicyCandidates.mode(modes, source, game, session.client);
        Mode chosen = super.chooseMode(modes, source, game);
        record(game, candidates, candidates.options.indexOf(chosen));
        return chosen;
    }

    @Override public TriggeredAbility chooseTriggeredAbility(List<TriggeredAbility> abilities, Game game) {
        TriggeredAbility chosen = super.chooseTriggeredAbility(abilities, game);
        if (live(game) && !abilities.isEmpty()) {
            record(game, PolicyCandidates.triggers(abilities, game, session.client), abilities.indexOf(chosen));
        }
        return chosen;
    }
}
