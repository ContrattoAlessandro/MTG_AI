package mage.player.ai.policy;

import mage.cards.Card;
import mage.constants.*;
import mage.game.Game;
import mage.game.events.*;
import mage.watchers.Watcher;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/** Captures sanitized events when they happen, including between actor decisions. */
public final class PolicyHistory extends Watcher {
    private final UUID observer;
    private final Map<String, Integer> cardIds;
    private final List<Map<String, Object>> pending = new ArrayList<>();
    // Shared by every copy: XMage restores a saved game state (and this watcher with
    // it) when an activation fails or a human undoes. Event numbers must keep
    // increasing, and events already sent to the policy must not be sent again.
    private final AtomicInteger sequence;
    private final AtomicInteger sent;
    private final Map<UUID, Integer> knownHand = new HashMap<>();

    public PolicyHistory(UUID observer, Map<String, Integer> cardIds) {
        super(WatcherScope.PLAYER);
        this.observer = observer;
        this.cardIds = new HashMap<>(cardIds);
        this.controllerId = observer;
        this.sequence = new AtomicInteger();
        this.sent = new AtomicInteger(-1);
    }

    private PolicyHistory(PolicyHistory other) {
        super(other);
        observer = other.observer;
        cardIds = new HashMap<>(other.cardIds);
        pending.addAll(other.pending);
        knownHand.putAll(other.knownHand);
        sequence = other.sequence;
        sent = other.sent;
    }

    @Override @SuppressWarnings("unchecked") public <T extends Watcher> T copy() {
        return (T) new PolicyHistory(this);
    }

    @Override
    public void reset() {
        // Watcher.reset is called at turn boundaries. History spans the whole game.
        super.reset();
    }

    @Override
    public void watch(GameEvent event, Game game) {
        if (game.isSimulation() || event instanceof BatchEvent) return;
        String type;
        UUID identity = event.getTargetId();
        boolean identityVisible = false;
        switch (event.getType()) {
            case DREW_CARD: type = "draw"; identityVisible = observer.equals(event.getPlayerId()); break;
            case SPELL_CAST: type = "cast"; identity = event.getSourceId(); identityVisible = true; break;
            case ACTIVATED_ABILITY: type = "activate"; identity = event.getSourceId(); identityVisible = true; break;
            case ATTACKER_DECLARED: type = "attack"; identityVisible = true; break;
            case BLOCKER_DECLARED: type = "block"; identityVisible = true; break;
            case DAMAGED_PLAYER: case DAMAGED_PERMANENT: type = "damage"; break;
            case GAINED_LIFE: case LOST_LIFE: type = "life"; break;
            case LIBRARY_SHUFFLED: type = "shuffle"; break;
            case ZONE_CHANGE:
                type = "zone_change";
                if (event instanceof ZoneChangeEvent) {
                    ZoneChangeEvent change = (ZoneChangeEvent) event;
                    // A library -> opponent hand transition reveals no identity.
                    identityVisible = publicZone(change.getToZone()) || publicZone(change.getFromZone())
                            || (change.getToZone() == Zone.HAND && observer.equals(event.getPlayerId()));
                }
                break;
            default: return;
        }
        Card card = identity == null ? null : game.getCard(identity);
        int cardId = 0;
        if (identityVisible && card != null && !card.isFaceDown(game)) {
            cardId = cardIds.getOrDefault(card.getName(), 0);
        }
        if (event instanceof ZoneChangeEvent) {
            ZoneChangeEvent change = (ZoneChangeEvent) event;
            if (change.getFromZone() == Zone.HAND && card != null && !observer.equals(card.getOwnerId())) {
                if (cardId == 0) knownHand.clear(); // an unrevealed departure cannot identify which known card left
                else knownHand.remove(identity);
            }
            if (change.getToZone() == Zone.HAND && publicZone(change.getFromZone())
                    && cardId > 0 && card != null && !observer.equals(card.getOwnerId())) {
                knownHand.put(identity, cardId);
            }
        }
        double[] features = new double[8];
        features[0] = event.getAmount() / 20.0;
        features[1] = game.getTurnNum() / 100.0;
        if (event instanceof ZoneChangeEvent) {
            ZoneChangeEvent change = (ZoneChangeEvent) event;
            features[2] = PolicyEncoder.zone(change.getFromZone()) / 7.0;
            features[3] = PolicyEncoder.zone(change.getToZone()) / 7.0;
        }
        pending.add(PolicyEncoder.map("seq", sequence.getAndIncrement(), "type", type,
            "actor", observer.equals(event.getPlayerId()) ? 0 : event.getPlayerId() == null ? 2 : 1,
            "card_id", cardId, "features", features));
    }

    private static boolean publicZone(Zone zone) {
        return zone == Zone.BATTLEFIELD || zone == Zone.GRAVEYARD || zone == Zone.STACK || zone == Zone.COMMAND;
        // Exile is conservative: face-down exile and owner-only visibility exist.
    }

    public List<Map<String, Object>> drain() {
        List<Map<String, Object>> result = new ArrayList<>();
        for (Map<String, Object> event : pending) {
            if ((Integer) event.get("seq") > sent.get()) result.add(event);
        }
        pending.clear();
        if (!result.isEmpty()) sent.set((Integer) result.get(result.size() - 1).get("seq"));
        return result;
    }

    /** An opponent hand card the observer was shown (e.g. by a choice); it stays known until it leaves the hand. */
    public void revealedInHand(UUID identity, int cardId) {
        if (cardId > 0) knownHand.put(identity, cardId);
    }

    public Map<Integer, Integer> knownHandCounts() {
        Map<Integer, Integer> counts = new HashMap<>();
        for (Integer id : knownHand.values()) counts.put(id, counts.getOrDefault(id, 0) + 1);
        return counts;
    }

    public void choice(int cardId, double[] actionFeatures, Game game) {
        double[] features = new double[8];
        System.arraycopy(actionFeatures, 0, features, 0, 8);
        pending.add(PolicyEncoder.map("seq", sequence.getAndIncrement(), "type", "choice", "actor", 0,
            "card_id", cardId, "features", features));
    }

    public void passed(UUID playerId, Game game) {
        double[] features = new double[8];
        features[1] = game.getTurnNum() / 100.0;
        pending.add(PolicyEncoder.map("seq", sequence.getAndIncrement(), "type", "pass", "actor",
            observer.equals(playerId) ? 0 : 1, "card_id", 0, "features", features));
    }
}
