package mage.player.ai.policy;

import com.google.gson.JsonObject;
import mage.MageObject;
import mage.Mana;
import mage.ConditionalMana;
import mage.abilities.*;
import mage.abilities.costs.Cost;
import mage.abilities.costs.SacrificeCost;
import mage.abilities.costs.mana.ManaCost;
import mage.abilities.costs.mana.ManaCosts;
import mage.abilities.effects.ContinuousEffect;
import mage.abilities.effects.ContinuousEffects;
import mage.abilities.effects.common.continuous.LookAtTopCardOfLibraryAnyTimeEffect;
import mage.abilities.mana.ActivatedManaAbilityImpl;
import mage.cards.Card;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.*;
import mage.counters.Counter;
import mage.game.ExileZone;
import mage.game.Game;
import mage.game.combat.CombatGroup;
import mage.game.events.GameEvent;
import mage.game.events.TableEvent;
import mage.game.permanent.Permanent;
import mage.game.permanent.PermanentToken;
import mage.game.stack.StackObject;
import mage.players.Player;
import mage.target.Target;
import mage.target.TargetAmount;
import java.util.*;

/**
 * A person playing in a browser (`mz web`). Every decision goes through the same
 * candidate lists as the policy (PolicyCandidates), so the person and the policy
 * have the same choices; the list, with readable labels and a snapshot of what
 * this player may see, is posted to the web server, which answers once the person
 * has clicked. Mana costs are paid by hand (one mana ability at a time) or by an
 * automatic payment the person asks for (see playMana).
 *
 * The web server holds each /engine/decide request for a while and answers
 * {"wait": true} when nobody has clicked yet; the player then asks again.
 * {"abort": true} ends the game (a new game or a concession in the browser).
 */
public final class WebPlayer extends PolicyPlayer {
    private static final int LOG_LINES = 400;
    // Above this many options a plain choice (a card name, a creature type) is searched.
    private static final int SEARCH_OPTIONS = 40;
    private final String web;
    // Shared by player copies, like the policy's session (state restores keep them).
    private transient PolicyClient bridge;
    private List<String> log = new ArrayList<>();
    private String[] prompt = {null};
    private String[] useTexts = {null, null};
    private String[] lastState = {""};
    // Cards revealed to everyone or looked at by this player (see watchSeen), newest last.
    private List<Map<String, Object>> seen = new ArrayList<>();
    private Set<String> seenNow = new HashSet<>();
    private int[] seenCount = {0};
    // The order the person chose for the triggered abilities being put on the stack.
    private List<UUID> triggerOrder = new ArrayList<>();
    // The ability whose mana cost is being paid automatically, and how deep in that
    // payment we are (mana abilities that cost mana pay their own cost inside it).
    private UUID[] autoPaying = {null};
    private int[] autoDepth = {0};
    // While paying automatically, the payment order group XMage's AI may tap (-1: any).
    private int[] payGroup = {-1};
    // The person cancelled a payment (PolicyPlayer then keeps offering the ability).
    private boolean[] cancelled = {false};
    // The permanent whose mana ability is being activated by hand, for the colour picker.
    private UUID[] manaSource = {null};
    // The kind of the next decision when it is not the candidates' own (a mana colour).
    private String[] askKind = {null};

    /**
     * policyEndpoint: the opponent policy's service, read for its card catalog only.
     * webEndpoint: the web server, e.g. http://127.0.0.1:8765/engine.
     */
    public WebPlayer(String name, RangeOfInfluence range, String policyEndpoint, String webEndpoint) {
        super(name, range, policyEndpoint, false, true);
        this.web = webEndpoint;
    }

    private WebPlayer(WebPlayer other) {
        super(other);
        this.web = other.web;
        this.bridge = other.bridge;
        this.log = other.log;
        this.prompt = other.prompt;
        this.useTexts = other.useTexts;
        this.lastState = other.lastState;
        this.seen = other.seen;
        this.seenNow = other.seenNow;
        this.seenCount = other.seenCount;
        this.triggerOrder = other.triggerOrder;
        this.autoPaying = other.autoPaying;
        this.autoDepth = other.autoDepth;
        this.payGroup = other.payGroup;
        this.cancelled = other.cancelled;
        this.manaSource = other.manaSource;
        this.askKind = other.askKind;
    }

    @Override public WebPlayer copy() { return new WebPlayer(this); }

    @Override public void initializeKnownDeck(Game game, UUID opponent) {
        super.initializeKnownDeck(game, opponent);
        openBridge(game);
    }

    /**
     * A seat at a table of any size (`mz lan`), against people and XMage's AI: there is no
     * policy, so no policy service is needed (policyEndpoint may be null).
     */
    public void joinTable(Game game) {
        startWithoutPolicy(game);
        openBridge(game);
        // At a table the others act between this player's priorities (a stack resolves after
        // the last pass, a bot casts on its turn): every board update and log line refreshes
        // this seat too, so all the browsers follow the same moment of the game.
        UUID me = getId();
        game.addTableEventListener(event -> {
            if (game.isSimulation()) return;
            TableEvent.EventType type = event.getEventType();
            if (type != TableEvent.EventType.UPDATE && type != TableEvent.EventType.INFO) return;
            Player player = game.getPlayer(me);
            if (player instanceof WebPlayer) ((WebPlayer) player).refresh(game);
        });
    }

    /**
     * Refreshes every person's board at a table: a priority changes hands without any table
     * event, and the page shows who holds it. A bot calls this at each of its priorities.
     */
    public static void refreshTable(Game game) { refreshOthers(game, null); }

    /** Refreshes the boards of the people at the table other than `except`. */
    private static void refreshOthers(Game game, UUID except) {
        if (game.isSimulation()) return;
        for (Player player : game.getState().getPlayers().values()) {
            if (player instanceof WebPlayer && !player.getId().equals(except)) ((WebPlayer) player).refresh(game);
        }
    }

    /** Sends the board for another player's move; this player's own replies wait for its next priority. */
    private void refresh(Game game) {
        try {
            post(game);
        } catch (RuntimeException ignored) {
            // cosmetic: a failed snapshot or a lost server shows up at the next decision
        }
    }

    private void openBridge(Game game) {
        // The server holds a decision request for at most about 20 s.
        bridge = new PolicyClient(web, false).readTimeout(120000);
        log = new ArrayList<>();
        seen = new ArrayList<>();
        seenNow = new HashSet<>();
        seenCount = new int[] {0};
        // XMage's own game log: the messages its client shows, already worded for
        // every player (hidden cards are not named).
        List<String> lines = log;
        int[] turn = {-1};
        game.addTableEventListener(event -> {
            if (!game.isSimulation()) watchSeen(game);
            if (event.getEventType() != TableEvent.EventType.INFO || event.getMessage() == null) return;
            synchronized (lines) {
                if (game.getTurnNum() != turn[0] && game.getActivePlayerId() != null) {
                    // XMage's log has no turn headers; the page shows these as separators.
                    turn[0] = game.getTurnNum();
                    Player active = game.getPlayer(game.getActivePlayerId());
                    lines.add("@turn " + turn[0] + " " + side(active.getId()) + " " + active.getName());
                }
                lines.add(event.getMessage());
                if (lines.size() > LOG_LINES) lines.remove(0);
            }
        });
    }

    /**
     * XMage keeps revealed cards (for everyone) and looked-at cards (per player) in
     * the game state only until the next board update: its client shows them as
     * windows then. Table events come before that clearing, so each one records the
     * groups that appeared since the previous event.
     */
    private void watchSeen(Game game) {
        Map<String, Cards> groups = new LinkedHashMap<>();
        for (Map.Entry<String, Cards> entry : game.getState().getRevealed().entrySet()) groups.put("reveal\u0000" + entry.getKey(), entry.getValue());
        // An any-time look at the top card (The Reality Chip, Mystic Forge) is renewed at every
        // effects update; the page shows that card on the library instead (library_top).
        Card top = null;
        for (Map.Entry<String, Cards> entry : game.getState().getLookedAt(getId()).entrySet()) {
            Cards cards = entry.getValue();
            if (cards != null && cards.size() == 1) {
                if (top == null) top = visibleTop(this, game);
                if (top != null && cards.contains(top.getId())) continue;
            }
            groups.put("look\u0000" + entry.getKey(), cards);
        }
        synchronized (seen) {
            Set<String> now = new HashSet<>();
            for (Map.Entry<String, Cards> entry : groups.entrySet()) {
                if (entry.getValue() == null || entry.getValue().isEmpty()) continue;
                List<UUID> ids = new ArrayList<>(entry.getValue());
                Collections.sort(ids);
                String key = entry.getKey() + ids;
                now.add(key);
                if (seenNow.contains(key)) continue;
                String[] parts = entry.getKey().split("\u0000", 2);
                List<Map<String, Object>> cards = new ArrayList<>();
                for (Card card : entry.getValue().getCards(game)) cards.add(card(card, game));
                seen.add(PolicyEncoder.map("id", ++seenCount[0], "kind", parts[0], "title", parts[1], "turn", game.getTurnNum(), "cards", cards));
                if (seen.size() > 20) seen.remove(0);
            }
            seenNow.clear();
            seenNow.addAll(now);
        }
    }

    // ---------------------------------------------------------------- decisions

    @Override protected int decide(Game game, List<Map<String, Object>> actions, List<?> options, Collection<Card> visible) {
        String question = prompt[0], asked = askKind[0];
        prompt[0] = null;
        askKind[0] = null;
        // A choice with one option is not asked, as for the policy. A priority with only
        // "pass" still goes to the server: in manual mode the person stops there (and sees
        // each stack object resolve); otherwise the server passes it at once.
        if (actions.size() == 1 && !"pass".equals(actions.get(0).get("type"))) return 0;
        List<Map<String, Object>> choices = new ArrayList<>();
        for (int i = 0; i < actions.size(); i++) {
            Map<String, Object> action = actions.get(i);
            Object option = options == null ? null : options.get(i);
            Map<String, Object> choice = PolicyEncoder.map("id", action.get("id"), "type", action.get("type"),
                "label", label(action, option, game), "source", action.get("source"), "target", action.get("target"));
            if (action.containsKey("mana")) choice.put("mana", true); // tap for mana at priority
            choices.add(choice);
        }
        String type = (String) actions.get(actions.size() - 1).get("type");
        Map<String, Object> state = snapshot(game, visible);
        Map<String, Object> request = PolicyEncoder.map("decision", UUID.randomUUID().toString(),
            "kind", asked != null ? asked : type, "prompt", question != null ? question : defaultPrompt(type, actions, game),
            "options", choices, "state", state);
        // A mana colour is chosen for the permanent the person tapped (the page shows it).
        if ("color".equals(asked) && manaSource[0] != null) request.put("source", manaSource[0].toString());
        String chosen = await(request, game);
        if (chosen == null) return 0; // conceded: the game ends at the next state check; the answer no longer matters
        for (int i = 0; i < actions.size(); i++) if (actions.get(i).get("id").equals(chosen)) return i;
        throw new IllegalStateException("browser returned an action outside the offered set: " + chosen);
    }

    /** The id of the option the person picked, or null after a concession. */
    private String await(Map<String, Object> request, Game game) {
        // While this person thinks, the others see the same moment (a new step, a spell being cast).
        refreshOthers(game, getId());
        while (true) {
            if (Thread.currentThread().isInterrupted()) throw new IllegalStateException("web game interrupted");
            JsonObject reply = bridge.request("POST", "/decide", request);
            if (reply.has("abort") && reply.get("abort").getAsBoolean()) throw new IllegalStateException("game abandoned in the browser");
            if (reply.has("concede") && reply.get("concede").getAsBoolean()) {
                game.concede(getId());
                return null;
            }
            if (reply.has("action_id")) return reply.get("action_id").getAsString();
            // nobody has clicked yet
        }
    }

    /**
     * Cards this player looks at while choosing among them, when none can be chosen
     * (Winota's six cards without a Human): the policy skips such a choice, a person
     * still sees the cards and confirms.
     */
    private void showCards(Game game, String text, Collection<Card> cards) {
        Map<String, Object> request = PolicyEncoder.map("decision", UUID.randomUUID().toString(), "kind", "look",
            "prompt", text, "options", Collections.singletonList(PolicyEncoder.map("id", "ok", "type", "look", "label", "OK",
                "source", null, "target", null)),
            "state", snapshot(game, cards));
        await(request, game);
    }

    // ----------------------------------------------------------------- combat

    /**
     * The whole declaration is asked at once (the policy asks one creature at a time):
     * one option per creature and defender it can attack; the person answers with the
     * ids of the pairs they chose, joined by commas (none: no attack).
     */
    @Override public void selectAttackers(Game game, UUID attackingPlayerId) {
        if (game.isSimulation()) { super.selectAttackers(game, attackingPlayerId); return; }
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_ATTACKERS_STEP_PRE, null, null, attackingPlayerId));
        if (game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_ATTACKERS, attackingPlayerId, attackingPlayerId))) return;
        List<UUID> defenders = new ArrayList<>(game.getCombat().getDefenders());
        Collections.sort(defenders);
        List<Map<String, Object>> options = new ArrayList<>();
        List<String> ready = new ArrayList<>();
        for (Permanent attacker : PolicyCandidates.attackers(this, game)) {
            ready.add(attacker.getId().toString());
            for (UUID defender : defenders) {
                if (!getAvailableAttackers(defender, game).contains(attacker)) continue;
                options.add(PolicyEncoder.map("id", attacker.getId() + ">" + defender, "type", "attack",
                    "label", attacker.getName() + " attacks " + name(defender.toString(), game),
                    "source", attacker.getId().toString(), "target", defender.toString()));
            }
        }
        if (!options.isEmpty()) {
            Map<String, Object> state = snapshot(game, null);
            state.put("can_attack", ready);
            Map<String, Object> request = PolicyEncoder.map("decision", UUID.randomUUID().toString(), "kind", "attackers",
                "prompt", "Declare attackers", "options", options, "state", state);
            for (UUID[] pair : pairs(await(request, game), options)) declareAttacker(pair[0], pair[1], game, false);
        }
        game.getPlayers().resetPassed();
    }

    /**
     * The whole declaration at once, as for attackers: one option per creature and
     * attacker it can block ("max": how many attackers it may block, 0 for any number).
     * XMage asks again when the blocks break a restriction (menace); the blocks of the
     * refused declaration are then removed and the person declares again.
     */
    @Override public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        if (game.isSimulation()) { super.selectBlockers(source, game, defendingPlayerId); return; }
        game.fireEvent(new GameEvent(GameEvent.EventType.DECLARE_BLOCKERS_STEP_PRE, null, null, defendingPlayerId));
        if (game.replaceEvent(GameEvent.getEvent(GameEvent.EventType.DECLARING_BLOCKERS, defendingPlayerId, defendingPlayerId))) return;
        boolean again = false;
        for (UUID blocker : new ArrayList<>(game.getCombat().getBlockers())) {
            Permanent permanent = game.getPermanent(blocker);
            if (permanent == null || !permanent.isControlledBy(getId())) continue;
            game.getCombat().removeBlocker(blocker, game);
            again = true;
        }
        // Only the attackers coming at this player (or its planeswalkers and battles): at a
        // table of more than two, the others are attacking someone else.
        List<UUID> attackers = new ArrayList<>();
        for (CombatGroup group : game.getCombat().getGroups()) {
            if (defendingPlayerId.equals(group.getDefendingPlayerId())) attackers.addAll(group.getAttackers());
        }
        Collections.sort(attackers);
        List<Map<String, Object>> options = new ArrayList<>();
        for (Permanent blocker : PolicyCandidates.blockers(this, game)) {
            for (UUID attacker : attackers) {
                if (!blocker.canBlock(attacker, game)) continue;
                options.add(PolicyEncoder.map("id", blocker.getId() + ">" + attacker, "type", "block",
                    "label", blocker.getName() + " blocks " + name(attacker.toString(), game),
                    "source", blocker.getId().toString(), "target", attacker.toString(), "max", blocker.getMaxBlocks()));
            }
        }
        if (!options.isEmpty()) {
            Map<String, Object> request = PolicyEncoder.map("decision", UUID.randomUUID().toString(), "kind", "blockers",
                "prompt", again ? "Those blocks are not legal (menace or another restriction): declare blockers again" : "Declare blockers",
                "options", options, "state", snapshot(game, null));
            Map<UUID, Integer> count = new HashMap<>();
            for (UUID[] pair : pairs(await(request, game), options)) {
                Permanent blocker = game.getPermanent(pair[0]);
                int limit = blocker == null ? 1 : blocker.getMaxBlocks();
                if (limit != 0 && count.merge(pair[0], 1, Integer::sum) > limit) throw new IllegalStateException("browser declared too many blocks for " + pair[0]);
                declareBlocker(defendingPlayerId, pair[0], pair[1], game);
            }
        }
        game.getPlayers().resetPassed();
    }

    /** The (creature, defender or attacker) pairs of a combat answer; none after a concession. */
    private static List<UUID[]> pairs(String answer, List<Map<String, Object>> options) {
        List<UUID[]> pairs = new ArrayList<>();
        if (answer == null) return pairs;
        Set<Object> offered = new HashSet<>();
        for (Map<String, Object> option : options) offered.add(option.get("id"));
        for (String id : answer.split(",")) {
            if (id.isEmpty()) continue;
            if (!offered.contains(id)) throw new IllegalStateException("browser returned a combat pair outside the offered set: " + id);
            String[] parts = id.split(">");
            pairs.add(new UUID[] {UUID.fromString(parts[0]), UUID.fromString(parts[1])});
        }
        return pairs;
    }

    @Override public boolean priority(Game game) {
        autoPaying[0] = null;
        cancelled[0] = false; // a payment cancelled outside priority (a "pay {1}?" trigger)
        // Each priority of this player, asked or not, refreshes the browser's board,
        // so the opponent's actions show up while they happen; at a table, the other people's
        // boards follow (who holds priority).
        if (!game.isSimulation()) {
            publish(game);
            refreshOthers(game, getId());
        }
        return super.priority(game);
    }

    @Override public void finish(Game game, boolean terminal, boolean truncated) {
        if (bridge == null) return;
        boolean everyoneLost = true, otherWon = false, othersLost = true;
        for (Player player : game.getPlayers().values()) {
            everyoneLost &= player.hasLost();
            if (player.getId().equals(getId())) continue;
            otherWon |= player.hasWon();
            othersLost &= player.hasLost();
        }
        String result = hasWon() ? "won" : hasLost() && everyoneLost ? "draw" : hasLost() ? "lost"
            : otherWon ? "lost" : othersLost ? "won" : truncated ? "turn limit" : "aborted";
        bridge.request("POST", "/finish", PolicyEncoder.map("result", result, "state", snapshot(game, null)));
    }

    private void publish(Game game) {
        JsonObject reply;
        try {
            reply = post(game);
        } catch (RuntimeException ignored) {
            return; // the board refresh is cosmetic; decisions report a lost server
        }
        if (reply == null) return;
        if (reply.has("abort") && reply.get("abort").getAsBoolean()) throw new IllegalStateException("game abandoned in the browser");
        if (reply.has("concede") && reply.get("concede").getAsBoolean()) game.concede(getId());
    }

    /** Posts this player's board unless it is the one last sent; the server's reply, or null. */
    private JsonObject post(Game game) {
        if (bridge == null) return null;
        Map<String, Object> state = snapshot(game, null);
        String key = String.valueOf(state);
        if (key.equals(lastState[0])) return null;
        lastState[0] = key;
        return bridge.request("POST", "/state", PolicyEncoder.map("state", state));
    }

    /**
     * XMage asks for one trigger at a time and puts it on the stack at once (its
     * targets are chosen in between). The person orders all of them in one go
     * instead, and the following calls are answered from that order; a trigger that
     * appears meanwhile asks again for the ones still waiting.
     */
    @Override public TriggeredAbility chooseTriggeredAbility(List<TriggeredAbility> abilities, Game game) {
        if (abilities.size() < 2 || game.isSimulation()) return super.chooseTriggeredAbility(abilities, game);
        Map<UUID, TriggeredAbility> waiting = new LinkedHashMap<>();
        for (TriggeredAbility ability : abilities) waiting.put(ability.getId(), ability);
        if (!triggerOrder.containsAll(waiting.keySet())) {
            triggerOrder.clear();
            triggerOrder.addAll(askTriggerOrder(abilities, game));
        }
        while (!triggerOrder.isEmpty()) {
            TriggeredAbility next = waiting.get(triggerOrder.remove(0));
            if (next != null) return next;
        }
        return abilities.get(0);
    }

    /** The abilities' ids in the order they go on the stack (the last one resolves first). */
    private List<UUID> askTriggerOrder(List<TriggeredAbility> abilities, Game game) {
        List<Map<String, Object>> options = new ArrayList<>();
        Set<String> ids = new HashSet<>();
        for (TriggeredAbility ability : abilities) {
            MageObject source = game.getObject(ability.getSourceId());
            options.add(PolicyEncoder.map("id", ability.getId().toString(), "type", "order", "label", ability.getRule(),
                "source", ability.getSourceId() == null ? null : ability.getSourceId().toString(), "target", null,
                "name", source == null ? "Ability" : source.getName()));
            ids.add(ability.getId().toString());
        }
        Map<String, Object> request = PolicyEncoder.map("decision", UUID.randomUUID().toString(), "kind", "triggers",
            "prompt", "Put your triggered abilities on the stack", "options", options, "state", snapshot(game, null));
        String answer = await(request, game);
        List<UUID> order = new ArrayList<>();
        if (answer == null) { // conceded
            for (TriggeredAbility ability : abilities) order.add(ability.getId());
            return order;
        }
        for (String id : answer.split(",")) if (!id.isEmpty()) order.add(UUID.fromString(id.trim()));
        Set<String> given = new HashSet<>();
        for (UUID id : order) given.add(id.toString());
        if (!given.equals(ids) || order.size() != ids.size()) throw new IllegalStateException("browser returned a trigger order that is not a permutation: " + answer);
        return order;
    }

    // ------------------------------------------------------------------- mana

    /**
     * Called by XMage until the cost is paid (mana already in the pool is used first),
     * and again for every mana ability that itself costs mana. The person taps one
     * source at a time (its mana ability; a colour is asked when it makes any colour),
     * asks for the automatic payment of the rest, or cancels the spell or ability.
     */
    @Override public boolean playMana(Ability ability, ManaCost unpaid, String promptText, Game game) {
        if (game.isSimulation()) return super.playMana(ability, unpaid, promptText, game);
        if (autoPaying[0] != null && (autoDepth[0] > 0 || autoPaying[0].equals(ability.getId()))) {
            if (autoPay(ability, unpaid, promptText, game)) return true;
            if (autoDepth[0] > 0) return false; // the outer automatic payment tries another source
        }
        boolean failed = autoPaying[0] != null && autoPaying[0].equals(ability.getId());
        autoPaying[0] = null;
        List<Map<String, Object>> options = new ArrayList<>();
        Map<String, ActivatedManaAbilityImpl> abilities = new HashMap<>();
        options.add(PolicyEncoder.map("id", "auto", "type", "mana", "label", "Auto pay", "source", null, "target", null));
        for (MageObject producer : manaProducers(game)) {
            Zone zone = game.getState().getZone(producer.getId());
            if (zone == null) continue;
            for (ActivatedManaAbilityImpl manaAbility : getUseableManaAbilities(producer, zone, game).values()) {
                if (!pays(manaAbility, ability, unpaid, game)) continue;
                String id = "mana-" + abilities.size();
                abilities.put(id, manaAbility);
                options.add(PolicyEncoder.map("id", id, "type", "mana", "label", manaAbility.getRule(),
                    "source", producer.getId().toString(), "target", null, "name", producer.getName()));
            }
        }
        options.add(PolicyEncoder.map("id", "cancel", "type", "mana", "label", "Cancel", "source", null, "target", null));
        String text = "Pay " + unpaid.getText() + from(ability, game);
        Map<String, Object> request = PolicyEncoder.map("decision", UUID.randomUUID().toString(), "kind", "mana",
            "prompt", failed ? "Auto pay could not finish. " + text : text, "options", options, "state", snapshot(game, null));
        String chosen = await(request, game);
        if (chosen == null) return false; // conceded
        if ("cancel".equals(chosen)) {
            cancelled[0] = true;
            return false;
        }
        if ("auto".equals(chosen)) {
            autoPaying[0] = ability.getId();
            return playMana(ability, unpaid, promptText, game); // asks again, saying so, when it cannot finish
        }
        ActivatedManaAbilityImpl manaAbility = abilities.get(chosen);
        if (manaAbility == null) throw new IllegalStateException("browser returned a mana option outside the offered set: " + chosen);
        manaSource[0] = manaAbility.getSourceId();
        try {
            activateAbility(manaAbility, game);
        } finally {
            manaSource[0] = null;
        }
        return true; // asked again until the cost is paid
    }

    /**
     * XMage's AI payment of the next part of the cost, offered one group of sources at
     * a time: artifacts, then lands, then other permanents (creatures), and last the
     * sources that can be sacrificed for mana (Treasure) or are not on the battlefield
     * (a card exiled from the hand). The first group that can pay that part pays it.
     * When none can, XMage's payment runs over every source (it also pays Phyrexian
     * mana with life and uses convoke-like special actions).
     */
    private boolean autoPay(Ability ability, ManaCost unpaid, String promptText, Game game) {
        ManaCost next = unpaid;
        if (unpaid instanceof ManaCosts && !((ManaCosts<?>) unpaid).isEmpty()) {
            ManaCosts<?> parts = (ManaCosts<?>) unpaid;
            next = parts.get(parts.size() - 1); // the part XMage's payment works on
        }
        int saved = payGroup[0];
        autoDepth[0]++;
        try {
            for (int group = 0; group < 4; group++) {
                payGroup[0] = -1;
                if (!groupPays(group, next, ability, game)) continue;
                payGroup[0] = group;
                if (super.playMana(ability, unpaid, promptText, game)) return true;
            }
            payGroup[0] = -1;
            return super.playMana(ability, unpaid, promptText, game);
        } finally {
            payGroup[0] = saved;
            autoDepth[0]--;
        }
    }

    private boolean groupPays(int group, ManaCost cost, Ability ability, Game game) {
        for (MageObject producer : manaProducers(game)) {
            Zone zone = game.getState().getZone(producer.getId());
            if (zone == null || payOrder(producer, game) != group) continue;
            for (ActivatedManaAbilityImpl manaAbility : getUseableManaAbilities(producer, zone, game).values()) {
                if (pays(manaAbility, ability, cost, game)) return true;
            }
        }
        return false;
    }

    // XMage's AI payment chooses among these (see autoPay).
    @Override public List<MageObject> getAvailableManaProducers(Game game) {
        List<MageObject> producers = super.getAvailableManaProducers(game);
        if (payGroup[0] >= 0) producers.removeIf(producer -> payOrder(producer, game) != payGroup[0]);
        return producers;
    }

    @Override public List<Permanent> getAvailableManaProducersWithCost(Game game) {
        List<Permanent> producers = super.getAvailableManaProducersWithCost(game);
        if (payGroup[0] >= 0) producers.removeIf(producer -> payOrder(producer, game) != payGroup[0]);
        return producers;
    }

    @Override protected boolean cancelledPayment() {
        boolean was = cancelled[0];
        cancelled[0] = false;
        return was;
    }

    /** The person may tap for mana at priority too; the pool empties between steps and phases as usual. */
    @Override protected boolean tapsForMana() { return true; }

    @Override public boolean activateAbility(ActivatedAbility ability, Game game) {
        // A mana ability chosen at priority (not while paying): its colour picker shows the source.
        if (!ability.isManaAbility() || manaSource[0] != null || game.isSimulation()) return super.activateAbility(ability, game);
        manaSource[0] = ability.getSourceId();
        try {
            return super.activateAbility(ability, game);
        } finally {
            manaSource[0] = null;
        }
    }

    private List<MageObject> manaProducers(Game game) {
        List<MageObject> producers = getAvailableManaProducers(game);
        producers.addAll(getAvailableManaProducersWithCost(game));
        producers.sort(Comparator.comparingInt((MageObject producer) -> payOrder(producer, game)).thenComparing(MageObject::getName));
        return producers;
    }

    /** Whether some mana this ability makes can go towards the unpaid cost. */
    private boolean pays(ActivatedManaAbilityImpl manaAbility, Ability ability, ManaCost unpaid, Game game) {
        List<Mana> made = manaAbility.getNetMana(game);
        if (made.isEmpty()) return true; // decided on activation
        for (Mana mana : made) {
            if (mana instanceof ConditionalMana && !((ConditionalMana) mana).apply(ability, game, getId(), unpaid)) continue;
            if (mana.getAny() > 0 || unpaid.testPay(mana)) return true;
        }
        return false;
    }

    /** 0 artifacts, 1 lands, 2 other permanents, 3 sources sacrificed for mana or not on the battlefield. */
    private int payOrder(MageObject producer, Game game) {
        if (!(producer instanceof Permanent)) return 3;
        for (ActivatedManaAbilityImpl ability : producer.getAbilities().getActivatedManaAbilities(Zone.BATTLEFIELD)) {
            for (Cost cost : ability.getCosts()) if (cost instanceof SacrificeCost) return 3;
        }
        return producer.isArtifact(game) ? 0 : producer.isLand(game) ? 1 : 2;
    }

    // ------------------------------------------------------- prompts and labels

    @Override protected boolean chooseTargets(Outcome outcome, Target target, Ability source, Game game, Cards fromCards) {
        prompt[0] = target.getMessage(game) + from(source, game);
        if (fromCards != null && !fromCards.isEmpty() && target.getTargets().isEmpty() && !game.isSimulation()
                && PolicyCandidates.target(this, target, source, game, fromCards, null).size() == 0) {
            String text = prompt[0];
            prompt[0] = null;
            showCards(game, "None of these can be chosen: " + Character.toLowerCase(text.charAt(0)) + text.substring(1), fromCards.getCards(game));
            return target.isChosen(game);
        }
        return super.chooseTargets(outcome, target, source, game, fromCards);
    }

    @Override public boolean chooseTargetAmount(Outcome outcome, TargetAmount target, Ability source, Game game) {
        prompt[0] = target.getMessage(game) + from(source, game);
        return super.chooseTargetAmount(outcome, target, source, game);
    }

    @Override public boolean chooseUse(Outcome outcome, String message, String secondMessage,
            String trueText, String falseText, Ability source, Game game) {
        prompt[0] = message + (secondMessage == null || secondMessage.isEmpty() ? "" : " " + secondMessage) + from(source, game);
        useTexts[0] = trueText;
        useTexts[1] = falseText;
        return super.chooseUse(outcome, message, secondMessage, trueText, falseText, source, game);
    }

    @Override public boolean choose(Outcome outcome, Choice choice, Game game) {
        // An automatic payment picks the colours it needs, as XMage's AI does.
        if (choice.isManaColorChoice() && autoPaying[0] != null) return chooseAsXMageAi(outcome, choice, game);
        String message = choice.getMessage();
        prompt[0] = message == null || message.isEmpty() ? "Make a choice" : message;
        if (choice.isManaColorChoice()) askKind[0] = "color";
        if (!choice.getKeyChoices().isEmpty() || choice.getChoices().size() <= SEARCH_OPTIONS || game.isSimulation()) {
            return super.choose(outcome, choice, game);
        }
        // "Name a card" offers every card name in XMage: the policy only gets the names of
        // its catalog, a person searches all of them. The cards of this game (both decks,
        // which the person chose) are suggested first.
        List<String> names = new ArrayList<>(choice.getChoices());
        Collections.sort(names);
        Set<String> inGame = new TreeSet<>();
        for (Card card : game.getCards()) if (choice.getChoices().contains(card.getName())) inGame.add(card.getName());
        String text = prompt[0];
        prompt[0] = null;
        Map<String, Object> request = PolicyEncoder.map("decision", UUID.randomUUID().toString(), "kind", "search",
            "prompt", text, "options", Collections.emptyList(), "names", names, "suggested", new ArrayList<>(inGame),
            "state", snapshot(game, null));
        String chosen = await(request, game);
        if (chosen == null) return false; // conceded
        if (!choice.getChoices().contains(chosen)) throw new IllegalStateException("browser returned a name outside the choice: " + chosen);
        choice.setChoice(chosen);
        return true;
    }

    @Override protected int chooseAmount(int min, int max, Game game, Ability source) {
        prompt[0] = "Choose a number from " + min + " to " + max + from(source, game);
        return super.chooseAmount(min, max, game, source);
    }

    @Override public Mode chooseMode(Modes modes, Ability source, Game game) {
        prompt[0] = "Choose a mode" + from(source, game);
        return super.chooseMode(modes, source, game);
    }

    private static String from(Ability source, Game game) {
        if (source == null) return "";
        MageObject object = game.getObject(source.getSourceId());
        return object == null ? "" : " (" + object.getName() + ")";
    }

    private String defaultPrompt(String type, List<Map<String, Object>> actions, Game game) {
        String source = name(actions.get(actions.size() - 1).get("source"), game);
        switch (type) {
            case "cast": case "activate": case "pass":
                return game.getActivePlayerId().equals(getId()) ? "Your turn: play a card or pass priority"
                    : capitalized(whose(game.getActivePlayerId(), game)) + "turn: respond or pass priority";
            case "mulligan": return "Keep this hand or mulligan?";
            case "attack": return "Attack with " + source + "?";
            case "block": return "Block with " + source + "?";
            case "order": return "Choose which goes on the stack (or applies) first";
            case "choose": return "Choose a pile";
            default: return "Choose";
        }
    }

    private String label(Map<String, Object> action, Object option, Game game) {
        String type = (String) action.get("type"), id = (String) action.get("id");
        Object text = action.get("text");
        String source = name(action.get("source"), game), target = name(action.get("target"), game);
        switch (type) {
            case "pass": return "Pass priority";
            case "cast":
                if (id.startsWith("cast-variant")) return String.valueOf(text);
                return "Cast " + source + cost(option);
            case "activate":
                if (option instanceof mage.abilities.PlayLandAbility) return "Play " + source;
                return source + ": " + text;
            case "target": return target == null ? "Done choosing" : target + where(action.get("target"), game);
            case "use": return Boolean.TRUE.equals(option) ? orElse(useTexts[0], "Yes") : orElse(useTexts[1], "No");
            case "mulligan": return "keep".equals(id) ? "Keep hand" : "Mulligan";
            case "amount": return String.valueOf(option);
            case "mode": return option == null ? "Done choosing modes" : String.valueOf(text);
            case "attack": return target == null ? "Don't attack" : "Attack " + target;
            case "block": return target == null ? ("none".equals(id) ? "Don't block" : "Done blocking") : "Block " + target;
            case "choose":
                if ("first".equals(id) || "second".equals(id)) return ("first".equals(id) ? "First pile" : "Second pile");
                return String.valueOf(text);
            default: return text == null ? type : String.valueOf(text);
        }
    }

    private static String cost(Object option) {
        if (!(option instanceof Ability)) return "";
        String text = ((Ability) option).getManaCostsToPay().getText();
        return text == null || text.isEmpty() ? "" : " " + text;
    }

    private static String orElse(String text, String fallback) {
        return text == null || text.isEmpty() ? fallback : text;
    }

    private String name(Object reference, Game game) {
        if (reference == null) return null;
        UUID id;
        try { id = UUID.fromString(reference.toString()); } catch (IllegalArgumentException e) { return reference.toString(); }
        Player player = game.getPlayer(id);
        if (player != null) return player.getId().equals(getId()) ? "yourself" : player.getName();
        MageObject object = game.getObject(id);
        if (object == null) return "?";
        if (object instanceof Card && ((Card) object).isFaceDown(game)) return "a face-down card";
        return object.getName();
    }

    private String where(Object reference, Game game) {
        UUID id = UUID.fromString(reference.toString());
        if (game.getPlayer(id) != null) return "";
        Zone zone = game.getState().getZone(id);
        Card card = game.getCard(id);
        Permanent permanent = game.getPermanent(id);
        UUID controller = permanent != null ? permanent.getControllerId() : card != null ? card.getOwnerId() : null;
        return zone == null ? "" : " (" + whose(controller, game) + zone.toString().toLowerCase(Locale.ROOT) + ")";
    }

    /** "your ", "opponent's " in a duel, or the player's name at a larger table ("Ann's "). */
    private String whose(UUID playerId, Game game) {
        if (playerId == null) return "";
        if (playerId.equals(getId())) return "your ";
        Player player = game.getPlayer(playerId);
        return player == null || game.getPlayers().size() <= 2 ? "opponent's " : player.getName() + "'s ";
    }

    private static String capitalized(String text) {
        return text.isEmpty() ? text : Character.toUpperCase(text.charAt(0)) + text.substring(1);
    }

    // ------------------------------------------------------------------- state

    private String side(UUID id) { return id == null ? null : id.equals(getId()) ? "you" : "opponent"; }

    private static String id(UUID id) { return id == null ? null : id.toString(); }

    /**
     * Every player in turn order, starting with this one (a duel: this player, then the
     * opponent), including those out of the game: GameState.getPlayerList leaves them out,
     * and the page still shows them (and the result once the game ends).
     */
    private List<Player> seating(Game game) {
        List<Player> all = new ArrayList<>(game.getState().getPlayers().values());
        // The turns follow XMage's PlayerList, which puts each joining player first: the
        // reverse of the players' map. Checked against the list, which also leaves out
        // the players out of the game.
        List<UUID> turns = new ArrayList<>(game.getState().getPlayerList());
        if (game.isTurnOrderReversed()) Collections.reverse(turns);
        List<Player> reversed = new ArrayList<>(all);
        Collections.reverse(reversed);
        if (follows(reversed, turns) && !follows(all, turns)) all = reversed;
        int me = 0;
        for (int i = 0; i < all.size(); i++) if (all.get(i).getId().equals(getId())) me = i;
        List<Player> seated = new ArrayList<>(all.subList(me, all.size()));
        seated.addAll(all.subList(0, me));
        return seated;
    }

    /** Whether `players`, without those missing from `turns`, go round in the order of `turns`. */
    private static boolean follows(List<Player> players, List<UUID> turns) {
        List<UUID> ids = new ArrayList<>();
        for (Player player : players) if (turns.contains(player.getId())) ids.add(player.getId());
        List<UUID> twice = new ArrayList<>(turns);
        twice.addAll(turns);
        return ids.size() == turns.size() && Collections.indexOfSubList(twice, ids) >= 0;
    }

    /**
     * Commander damage dealt to `player` by each other player's commanders (the highest per
     * commander, by the id of the commander's owner); 21 from one commander loses.
     */
    private static Map<String, Object> commanderDamageTo(Player player, List<Player> seated, Game game) {
        Map<String, Object> taken = new LinkedHashMap<>();
        for (Player other : seated) {
            if (other == player) continue;
            int most = 0;
            for (UUID commander : game.getCommandersIds(other, CommanderCardType.ANY, false)) {
                mage.watchers.common.CommanderInfoWatcher watcher =
                    game.getState().getWatcher(mage.watchers.common.CommanderInfoWatcher.class, commander);
                if (watcher != null) most = Math.max(most, watcher.getDamageToPlayer().getOrDefault(player.getId(), 0));
            }
            taken.put(other.getId().toString(), most);
        }
        return taken;
    }

    /**
     * What this player may see: every board, public zones, the own hand, hidden counts.
     * Players and objects carry their side ("you" or "opponent", as a duel shows them)
     * and, for tables of more than two, the id of the player they belong to.
     */
    private Map<String, Object> snapshot(Game game, Collection<Card> visible) {
        List<Map<String, Object>> players = new ArrayList<>();
        Player me = game.getPlayer(getId());
        List<Player> seated = seating(game);
        for (Player player : seated) {
            List<Map<String, Object>> graveyard = new ArrayList<>(), exile = new ArrayList<>(), command = new ArrayList<>();
            for (Card card : player.getGraveyard().getCards(game)) graveyard.add(card(card, game));
            for (ExileZone zone : game.getExile().getExileZones()) {
                for (Card card : zone.getCards(game)) {
                    if (card.getOwnerId().equals(player.getId()) && !card.isFaceDown(game)) exile.add(card(card, game));
                }
            }
            for (UUID commander : game.getCommandersIds(player, CommanderCardType.ANY, false)) {
                if (game.getState().getZone(commander) == Zone.COMMAND) command.add(card(game.getCard(commander), game));
            }
            Map<String, Object> commanderDamageBy = commanderDamageTo(player, seated, game);
            int commanderDamage = 0;
            for (Object taken : commanderDamageBy.values()) commanderDamage = Math.max(commanderDamage, (Integer) taken);
            mage.players.ManaPool pool = player.getManaPool();
            Card top = visibleTop(player, game);
            players.add(PolicyEncoder.map("id", player.getId().toString(), "side", side(player.getId()),
                "name", player.getName(), "life", player.getLife(),
                "poison", player.getCountersCount(mage.counters.CounterType.POISON),
                "hand", player.getHand().size(), "library", player.getLibrary().size(),
                "library_top", top == null ? Collections.emptyList() : Collections.singletonList(card(top, game)),
                "commander_damage", commanderDamage, "commander_damage_by", commanderDamageBy,
                "mana", PolicyEncoder.map("W", pool.getWhite(), "U", pool.getBlue(), "B", pool.getBlack(),
                    "R", pool.getRed(), "G", pool.getGreen(), "C", pool.getColorless()),
                "graveyard", graveyard, "exile", exile, "command", command,
                "won", player.hasWon(), "lost", player.hasLost()));
        }
        List<Map<String, Object>> battlefield = new ArrayList<>();
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) battlefield.add(card(permanent, game));
        List<Map<String, Object>> hand = new ArrayList<>();
        for (Card card : me.getHand().getCards(game)) hand.add(card(card, game));
        List<Map<String, Object>> stack = new ArrayList<>();
        for (StackObject object : game.getStack()) {
            Map<String, Object> entry = card(object, game);
            Ability ability = object.getStackAbility();
            if (object instanceof mage.game.stack.StackAbility) {
                // XMage names abilities "stack ability (rules)"; show the source's name.
                MageObject source = game.getObject(object.getSourceId());
                entry.put("name", source == null ? "Ability" : source.getName());
                entry.put("ability", true);
            }
            Map<String, Map<String, Object>> links = new LinkedHashMap<>();
            if (ability != null) {
                entry.put("rules", Collections.singletonList(ability.getRule()));
                AbilityType kind = ability.getAbilityType();
                entry.put("ability_kind", kind == null ? null : kind.isTriggeredAbility() ? "triggered" : kind.isActivatedAbility() ? "activated" : null);
                // Every selected mode: its targets, then what its effects point at
                // without targeting ("that creature", a card exiled earlier).
                for (UUID modeId : ability.getModes().getSelectedModes()) {
                    Mode mode = ability.getModes().get(modeId);
                    if (mode == null) continue;
                    for (Target target : mode.getTargets()) {
                        for (UUID id : target.getTargets()) link(links, id, "target", game);
                    }
                    for (mage.abilities.effects.Effect effect : mode.getEffects()) {
                        try {
                            for (UUID id : effect.getTargetPointer().getTargets(game, ability)) link(links, id, "related", game);
                        } catch (RuntimeException ignored) {
                            // some pointers only resolve while the effect applies
                        }
                    }
                }
            }
            // The object that put this on the stack (a spell is its own source).
            if (object instanceof mage.game.stack.StackAbility && object.getSourceId() != null) {
                Map<String, Object> source = place(object.getSourceId(), game);
                links.remove(object.getSourceId().toString());
                entry.put("source", source);
            }
            links.remove(object.getId().toString());
            entry.put("targets", new ArrayList<>(links.values()));
            entry.put("controller", side(object.getControllerId()));
            entry.put("controller_id", id(object.getControllerId()));
            stack.add(entry);
        }
        List<Map<String, Object>> combat = new ArrayList<>();
        for (CombatGroup group : game.getCombat().getGroups()) {
            List<String> attackers = new ArrayList<>(), blockers = new ArrayList<>();
            for (UUID id : group.getAttackers()) attackers.add(id.toString());
            for (UUID id : group.getBlockers()) blockers.add(id.toString());
            combat.add(PolicyEncoder.map("attackers", attackers, "blockers", blockers,
                "defender", id(group.getDefenderId()), "defending_player", id(group.getDefendingPlayerId())));
        }
        // Cards this choice lets the player look at (a library search, a revealed hand).
        List<Map<String, Object>> revealed = new ArrayList<>();
        if (visible != null) for (Card card : visible) revealed.add(card(card, game));
        List<String> lines;
        synchronized (log) { lines = new ArrayList<>(log); }
        PhaseStep step = game.getTurnStepType();
        List<Map<String, Object>> seenGroups;
        synchronized (seen) { seenGroups = new ArrayList<>(seen); }
        return PolicyEncoder.map("turn", game.getTurnNum(), "step", step == null ? "" : step.name(),
            "active", side(game.getActivePlayerId()), "priority", side(game.getState().getPriorityPlayerId()),
            "active_id", id(game.getActivePlayerId()), "priority_id", id(game.getState().getPriorityPlayerId()),
            "players", players, "battlefield", battlefield, "hand", hand, "stack", stack, "combat", combat,
            "revealed", revealed, "seen", seenGroups, "log", lines);
    }

    private void link(Map<String, Map<String, Object>> links, UUID id, String kind, Game game) {
        if (id == null || links.containsKey(id.toString())) return;
        Map<String, Object> place = place(id, game);
        place.put("kind", kind);
        links.put(id.toString(), place);
    }

    /**
     * Where an object a stack item refers to is, so the page can point at it even
     * when it is not on the battlefield: a player, or a zone and whose zone it is.
     * Names of cards in hidden zones are not given.
     */
    private Map<String, Object> place(UUID id, Game game) {
        Map<String, Object> place = PolicyEncoder.map("id", id.toString());
        Player player = game.getPlayer(id);
        if (player != null) {
            place.put("zone", "player");
            place.put("owner", side(id));
            place.put("owner_id", id.toString());
            return place;
        }
        Zone zone = game.getState().getZone(id);
        place.put("zone", zone == null ? null : zone == Zone.EXILED ? "exile" : zone.name().toLowerCase(Locale.ROOT));
        Permanent permanent = game.getPermanent(id);
        Card card = game.getCard(id);
        StackObject spell = game.getStack().getStackObject(id);
        UUID owner = permanent != null ? permanent.getControllerId() : spell != null ? spell.getControllerId() : card != null ? card.getOwnerId() : null;
        place.put("owner", side(owner));
        place.put("owner_id", id(owner));
        boolean hidden = zone == Zone.LIBRARY || zone == Zone.HAND && !getId().equals(owner);
        MageObject object = game.getObject(id);
        if (!hidden && object != null) place.put("name", object instanceof Card && ((Card) object).isFaceDown(game) ? "Face-down card" : object.getName());
        return place;
    }

    /**
     * The top card of `player`'s library when this player may see it: played revealed
     * (Courser of Kruphix), or this player's own any-time look (The Reality Chip, Mystic Forge).
     */
    private Card visibleTop(Player player, Game game) {
        Card top = player.getLibrary().getFromTop(game);
        if (top == null) return null;
        if (player.isTopCardRevealed()) return top;
        if (!getId().equals(player.getId())) return null;
        ContinuousEffects effects = game.getContinuousEffects();
        for (ContinuousEffect effect : effects.getLayeredEffects(game)) {
            if (!(effect instanceof LookAtTopCardOfLibraryAnyTimeEffect)) continue;
            if (!((LookAtTopCardOfLibraryAnyTimeEffect) effect).canLookAtNextTopLibraryCard(game)) continue;
            for (Ability ability : effects.getLayeredEffectAbilities(effect)) {
                if (getId().equals(ability.getControllerId())) return top;
            }
        }
        return null;
    }

    private Map<String, Object> card(MageObject object, Game game) {
        boolean faceDown = object instanceof Card && ((Card) object).isFaceDown(game);
        Map<String, Object> view = PolicyEncoder.map("id", object.getId().toString());
        // A face-down permanent is shown as such, except to its controller.
        boolean concealed = faceDown && !(object instanceof Permanent && getId().equals(((Permanent) object).getControllerId()));
        view.put("name", concealed ? "Face-down card" : object.getName());
        view.put("face_down", faceDown);
        view.put("token", object instanceof PermanentToken);
        if (!concealed) {
            List<String> types = new ArrayList<>();
            for (CardType type : object.getCardType(game)) types.add(type.toString());
            String subtypes = object.getSubtype(game).toString();
            view.put("types", String.join(" ", types) + (subtypes.isEmpty() || "[]".equals(subtypes) ? "" : " — " + subtypes.replaceAll("[\\[\\],]", "")));
            view.put("cost", String.join("", object.getManaCostSymbols()));
            List<String> rules = object instanceof Card ? ((Card) object).getRules(game) : new ArrayList<>();
            view.put("rules", rules);
        }
        view.put("creature", object.isCreature(game));
        view.put("land", object.isLand(game));
        if (object.isCreature(game)) {
            view.put("power", object.getPower().getValue());
            view.put("toughness", object.getToughness().getValue());
        }
        if (object instanceof Permanent) {
            Permanent permanent = (Permanent) object;
            view.put("controller", side(permanent.getControllerId()));
            view.put("controller_id", id(permanent.getControllerId()));
            view.put("tapped", permanent.isTapped());
            view.put("phased_out", !permanent.isPhasedIn());
            view.put("sick", permanent.isCreature(game) && permanent.hasSummoningSickness());
            view.put("damage", permanent.getDamage());
            view.put("attacking", permanent.isAttacking());
            view.put("blocking", permanent.getBlocking() > 0);
            view.put("attached_to", permanent.getAttachedTo() == null ? null : permanent.getAttachedTo().toString());
            Map<String, Object> counters = new TreeMap<>();
            for (Counter counter : permanent.getCounters(game).values()) counters.put(counter.getName(), counter.getCount());
            view.put("counters", counters);
        }
        return view;
    }
}
