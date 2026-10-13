package mage.player.ai.policy;

import mage.game.Game;
import java.util.UUID;

/** A player whose decisions are recorded through a policy service: the policy itself, or the XMage-AI teacher. */
public interface RecordingPlayer {
    /** Reads the announced opponent deck list; call before game.start. */
    void initializeKnownDeck(Game game, UUID opponent);

    /**
     * terminal: the game reached a real result. truncated: an error-free game
     * stopped by the turn limit; it is recorded with result 0 and bootstrapped by
     * the learner, never scored as a draw. Neither: aborted, trajectory discarded.
     */
    void finish(Game game, boolean terminal, boolean truncated);
}
