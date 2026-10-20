package mage.player.ai;

import mage.abilities.Ability;
import mage.abilities.ActivatedAbility;
import mage.abilities.SpellAbility;
import mage.abilities.common.PassAbility;
import mage.abilities.costs.mana.ManaCost;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import org.apache.log4j.Logger;

import java.lang.reflect.Field;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * XMage's Mad AI (ComputerPlayer7) as an opponent and teacher, with three fixes that
 * need no change to XMage itself (only its public player API):
 * - it stops activating the same ability after 50 activations in one turn
 *   (-Dxmage.ai.maxActivationsPerTurn): its search otherwise repeats a real infinite
 *   loop (Basalt Monolith untapping with Kinnan) until the game times out;
 * - nested mana payments stop at depth 8 (mana abilities that cost mana paying for
 *   each other overflowed the stack);
 * - its copies keep the fast-fail setting, which PlayerImpl's copy constructor drops.
 */
public class XMageAiPlayer extends ComputerPlayer7 {

    private static final Logger logger = Logger.getLogger(XMageAiPlayer.class);
    static final int MAX_SAME_ACTIVATIONS_PER_TURN = Math.max(1, Integer.getInteger("xmage.ai.maxActivationsPerTurn", 50));
    private static final int MAX_MANA_PAYMENT_DEPTH = 8;

    /** Shared by every copy: XMage restores saved player copies during rules processing. */
    private static final class Activations {
        final Map<String, Integer> counts = new ConcurrentHashMap<>();
        int priorityDepth;
        boolean stopped;   // cap reached in this priority: pass, activate nothing more
    }

    private final Activations activations;
    private transient int manaPaymentDepth = 0;

    public XMageAiPlayer(String name, RangeOfInfluence range, int skill) {
        super(name, range, skill);
        this.activations = new Activations();
    }

    protected XMageAiPlayer(XMageAiPlayer other) {
        super(other);
        setFastFailInTestMode(other.isFastFailInTestMode());
        this.activations = other.activations;
    }

    @Override public XMageAiPlayer copy() { return new XMageAiPlayer(this); }

    @Override public boolean priority(Game game) {
        if (game.isSimulation()) return super.priority(game);
        activations.priorityDepth++;
        activations.stopped = false;
        try {
            return super.priority(game);
        } finally {
            activations.priorityDepth--;
        }
    }

    @Override public boolean activateAbility(ActivatedAbility ability, Game game) {
        if (overActivationCap(ability, game)) {
            stopActing(ability, game);
            return false;
        }
        return super.activateAbility(ability, game);
    }

    /** The activation itself, bypassing the cap (for a subclass that checked it first). */
    protected final boolean activateUnchecked(ActivatedAbility ability, Game game) {
        return super.activateAbility(ability, game);
    }

    /** Counts an activation the AI chose at priority; true when it must not happen. */
    protected boolean overActivationCap(ActivatedAbility ability, Game game) {
        if (game.isSimulation() || activations.priorityDepth == 0 || ability.isManaAbility()) return false;
        if (activations.stopped) return true;
        if (ability instanceof SpellAbility || ability instanceof PassAbility) return false;
        if (activations.counts.size() > 1024) activations.counts.clear();
        String key = game.getTurnNum() + "|" + ability.getSourceId() + "|" + ability.getOriginalId();
        return activations.counts.merge(key, 1, Integer::sum) > MAX_SAME_ACTIVATIONS_PER_TURN;
    }

    /** Passes instead of the capped activation; the rest of this priority's plan is skipped. */
    protected void stopActing(ActivatedAbility ability, Game game) {
        if (activations.stopped) return;
        activations.stopped = true;
        logger.warn("AI activation cap reached, passing: " + ability + " (" + getName() + ")");
        pass(game);
    }

    @Override public boolean playMana(Ability ability, ManaCost unpaid, String promptText, Game game) {
        if (manaPaymentDepth >= MAX_MANA_PAYMENT_DEPTH) return false;
        manaPaymentDepth++;
        try {
            return super.playMana(ability, unpaid, promptText, game);
        } finally {
            manaPaymentDepth--;
        }
    }

    /**
     * XMage's AI players share one pool of search threads (5). With more concurrent
     * games than that, their searches queue and hit the think-time cap. Resizes the
     * pool; false (with a warning) if this XMage version keeps it elsewhere, in which
     * case the default size stays and games still work.
     */
    public static boolean configureSimulationThreads(int threads) {
        try {
            Field field = ComputerPlayer6.class.getDeclaredField("threadPoolSimulations");
            field.setAccessible(true);
            ExecutorService pool = (ExecutorService) field.get(null);
            if (!(pool instanceof ThreadPoolExecutor)) throw new IllegalStateException(String.valueOf(pool));
            ThreadPoolExecutor executor = (ThreadPoolExecutor) pool;
            int size = Math.max(1, threads);
            if (size > executor.getMaximumPoolSize()) {
                executor.setMaximumPoolSize(size);
                executor.setCorePoolSize(size);
            } else {
                executor.setCorePoolSize(size);
                executor.setMaximumPoolSize(size);
            }
            return true;
        } catch (ReflectiveOperationException | RuntimeException error) {
            logger.warn("could not resize XMage's AI simulation thread pool; keeping its default: " + error);
            return false;
        }
    }
}
