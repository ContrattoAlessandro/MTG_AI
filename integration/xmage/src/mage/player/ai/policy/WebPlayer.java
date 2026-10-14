package mage.player.ai.policy;

import com.google.gson.JsonObject;
import mage.MageObject;
import mage.abilities.*;
import mage.cards.Card;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.*;
import mage.counters.Counter;
import mage.game.ExileZone;
import mage.game.Game;
import mage.game.combat.CombatGroup;
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
 * has clicked. Mana is paid by XMage's automatic payment, as for the policy.
 *
 * The web server holds each /engine/decide request for a while and answers
 * {"wait": true} when nobody has clicked yet; the player then asks again.
 * {"abort": true} ends the game (a new game or a concession in the browser).
 */
public final class WebPlayer extends PolicyPlayer {
    private static final int LOG_LINES = 400;
    private final String web;
    // Shared by player copies, like the policy's session (state restores keep them).
    private transient PolicyClient bridge;
    private List<String> log = new ArrayList<>();
    private String[] prompt = {null};
    private String[] useTexts = {null, null};
    private String[] lastState = {""};

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
    }

    @Override public WebPlayer copy() { return new WebPlayer(this); }

    @Override public void initializeKnownDeck(Game game, UUID opponent) {
        super.initializeKnownDeck(game, opponent);
        // The server holds a decision request for at most about 20 s.
        bridge = new PolicyClient(web, false).readTimeout(120000);
        log = new ArrayList<>();
        // XMage's own game log: the messages its client shows, already worded for
        // every player (hidden cards are not named).
        List<String> lines = log;
        int[] turn = {-1};
        game.addTableEventListener(event -> {
            if (event.getEventType() != TableEvent.EventType.INFO || event.getMessage() == null) return;
            synchronized (lines) {
                if (game.getTurnNum() != turn[0] && game.getActivePlayerId() != null) {
                    // XMage's log has no turn headers; the page shows these as separators.
                    turn[0] = game.getTurnNum();
                    Player active = game.getPlayer(game.getActivePlayerId());
                    lines.add("@turn " + turn[0] + " " + (getId().equals(active.getId()) ? "you" : "opponent"));
                }
                lines.add(event.getMessage());
                if (lines.size() > LOG_LINES) lines.remove(0);
            }
        });
    }

    // ---------------------------------------------------------------- decisions

    @Override protected int decide(Game game, List<Map<String, Object>> actions, List<?> options, Collection<Card> visible) {
        String question = prompt[0];
        prompt[0] = null;
        // A choice with one option is not asked, as for the policy.
        if (actions.size() == 1) return 0;
        List<Map<String, Object>> choices = new ArrayList<>();
        for (int i = 0; i < actions.size(); i++) {
            Map<String, Object> action = actions.get(i);
            Object option = options == null ? null : options.get(i);
            choices.add(PolicyEncoder.map("id", action.get("id"), "type", action.get("type"),
                "label", label(action, option, game), "source", action.get("source"), "target", action.get("target")));
        }
        String type = (String) actions.get(actions.size() - 1).get("type");
        Map<String, Object> request = PolicyEncoder.map("decision", UUID.randomUUID().toString(),
            "kind", type, "prompt", question != null ? question : defaultPrompt(type, actions, game),
            "options", choices, "state", snapshot(game, visible));
        while (true) {
            if (Thread.currentThread().isInterrupted()) throw new IllegalStateException("web game interrupted");
            JsonObject reply = bridge.request("POST", "/decide", request);
            if (reply.has("abort") && reply.get("abort").getAsBoolean()) throw new IllegalStateException("game abandoned in the browser");
            if (reply.has("concede") && reply.get("concede").getAsBoolean()) {
                game.concede(getId()); // the game ends at the next state check; the answer no longer matters
                return 0;
            }
            if (!reply.has("action_id")) continue; // nobody has clicked yet
            String chosen = reply.get("action_id").getAsString();
            for (int i = 0; i < actions.size(); i++) if (actions.get(i).get("id").equals(chosen)) return i;
            throw new IllegalStateException("browser returned an action outside the offered set: " + chosen);
        }
    }

    @Override public boolean priority(Game game) {
        // Each priority of this player, asked or not, refreshes the browser's board,
        // so the opponent's actions show up while they happen.
        if (!game.isSimulation()) publish(game);
        return super.priority(game);
    }

    @Override public void finish(Game game, boolean terminal, boolean truncated) {
        if (bridge == null) return;
        Player opponent = game.getPlayer(game.getOpponents(getId()).iterator().next());
        String result = hasWon() ? "won" : hasLost() && opponent.hasLost() ? "draw" : hasLost() ? "lost"
            : opponent.hasWon() ? "lost" : opponent.hasLost() ? "won" : truncated ? "turn limit" : "aborted";
        bridge.request("POST", "/finish", PolicyEncoder.map("result", result, "state", snapshot(game, null)));
    }

    private void publish(Game game) {
        Map<String, Object> state = snapshot(game, null);
        String key = String.valueOf(state);
        if (key.equals(lastState[0])) return;
        lastState[0] = key;
        JsonObject reply;
        try {
            reply = bridge.request("POST", "/state", PolicyEncoder.map("state", state));
        } catch (RuntimeException ignored) {
            return; // the board refresh is cosmetic; decisions report a lost server
        }
        if (reply.has("abort") && reply.get("abort").getAsBoolean()) throw new IllegalStateException("game abandoned in the browser");
        if (reply.has("concede") && reply.get("concede").getAsBoolean()) game.concede(getId());
    }

    // ------------------------------------------------------- prompts and labels

    @Override protected boolean makeChoice(Outcome outcome, Target target, Ability source, Game game, Cards fromCards) {
        prompt[0] = target.getMessage(game) + from(source, game);
        return super.makeChoice(outcome, target, source, game, fromCards);
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
        String message = choice.getMessage();
        prompt[0] = message == null || message.isEmpty() ? "Make a choice" : message;
        return super.choose(outcome, choice, game);
    }

    @Override protected int makeChoiceAmount(int min, int max, Game game, Ability source, boolean isManaPay) {
        prompt[0] = "Choose a number from " + min + " to " + max + from(source, game);
        return super.makeChoiceAmount(min, max, game, source, isManaPay);
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
                    : "Opponent's turn: respond or pass priority";
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
        String whose = controller == null ? "" : controller.equals(getId()) ? "your " : "opponent's ";
        return zone == null ? "" : " (" + whose + zone.toString().toLowerCase(Locale.ROOT) + ")";
    }

    // ------------------------------------------------------------------- state

    private String side(UUID id) { return id == null ? null : id.equals(getId()) ? "you" : "opponent"; }

    /** What this player may see: both boards, public zones, the own hand, hidden counts. */
    private Map<String, Object> snapshot(Game game, Collection<Card> visible) {
        List<Map<String, Object>> players = new ArrayList<>();
        Player me = game.getPlayer(getId());
        Player enemy = game.getPlayer(game.getOpponents(getId()).iterator().next());
        for (Player player : Arrays.asList(me, enemy)) {
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
            int commanderDamage = 0;
            Player other = player == me ? enemy : me;
            for (UUID commander : game.getCommandersIds(other, CommanderCardType.ANY, false)) {
                mage.watchers.common.CommanderInfoWatcher watcher =
                    game.getState().getWatcher(mage.watchers.common.CommanderInfoWatcher.class, commander);
                if (watcher != null) commanderDamage = Math.max(commanderDamage, watcher.getDamageToPlayer().getOrDefault(player.getId(), 0));
            }
            mage.players.ManaPool pool = player.getManaPool();
            players.add(PolicyEncoder.map("id", player.getId().toString(), "side", side(player.getId()),
                "name", player.getName(), "life", player.getLife(),
                "poison", player.getCountersCount(mage.counters.CounterType.POISON),
                "hand", player.getHand().size(), "library", player.getLibrary().size(),
                "commander_damage", commanderDamage,
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
            List<String> targets = new ArrayList<>();
            if (ability != null) {
                entry.put("rules", Collections.singletonList(ability.getRule()));
                for (Target target : ability.getTargets()) {
                    for (UUID id : target.getTargets()) targets.add(id.toString());
                }
            }
            entry.put("targets", targets);
            entry.put("controller", side(object.getControllerId()));
            stack.add(entry);
        }
        List<Map<String, Object>> combat = new ArrayList<>();
        for (CombatGroup group : game.getCombat().getGroups()) {
            List<String> attackers = new ArrayList<>(), blockers = new ArrayList<>();
            for (UUID id : group.getAttackers()) attackers.add(id.toString());
            for (UUID id : group.getBlockers()) blockers.add(id.toString());
            combat.add(PolicyEncoder.map("attackers", attackers, "blockers", blockers,
                "defender", group.getDefenderId() == null ? null : group.getDefenderId().toString()));
        }
        // Cards this choice lets the player look at (a library search, a revealed hand).
        List<Map<String, Object>> revealed = new ArrayList<>();
        if (visible != null) for (Card card : visible) revealed.add(card(card, game));
        List<String> lines;
        synchronized (log) { lines = new ArrayList<>(log); }
        PhaseStep step = game.getTurnStepType();
        return PolicyEncoder.map("turn", game.getTurnNum(), "step", step == null ? "" : step.name(),
            "active", side(game.getActivePlayerId()), "priority", side(game.getState().getPriorityPlayerId()),
            "players", players, "battlefield", battlefield, "hand", hand, "stack", stack, "combat", combat,
            "revealed", revealed, "log", lines);
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
