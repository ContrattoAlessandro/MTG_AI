package org.mage.magezero;

import com.google.gson.GsonBuilder;
import mage.cards.decks.*;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.*;
import mage.constants.*;
import mage.game.*;
import mage.game.match.*;
import mage.game.mulligan.MulliganType;
import mage.player.ai.ComputerPlayerMCTS2;
import mage.player.ai.ComputerPlayerMCTS;
import mage.player.ai.encoder.StateEncoder;
import mage.player.ai.policy.PolicyPlayer;
import mage.players.Player;
import mage.util.RandomUtil;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;

/** Fixed-opponent evaluation, with game outcomes and explicitly bounded search. */
public final class PolicyArena {
    // The upstream root copies actual hidden hands; perfectInfo=false only
    // sanitizes its neural features. Use its existing sampler at the root for
    // a public-information comparison. The search/heuristic itself is unchanged.
    private static final class PublicMCTS extends ComputerPlayerMCTS2 {
        PublicMCTS(String name) { super(name, RangeOfInfluence.ONE, 6); }
        private PublicMCTS(PublicMCTS other) { super(other); }
        @Override public PublicMCTS copy() { return new PublicMCTS(this); }
        @Override protected Game createMCTSGame(Game game) {
            Game simulation = super.createMCTSGame(game);
            ComputerPlayerMCTS.shuffleUnknowns(simulation, getId());
            return simulation;
        }
    }

    public static void main(String[] args) throws Exception {
        Map<String,String> options = new HashMap<>();
        if (args.length % 2 != 0) throw new IllegalArgumentException("arguments require --key value pairs");
        for (int i=0;i<args.length;i+=2) options.put(args[i], args[i+1]);
        RepositoryUtil.bootstrapLocalDb(); CardScanner.scan();
        DeckCardLists deck = DeckImporter.importDeckFromFile(options.get("--deck"), true);
        int games = Integer.parseInt(options.getOrDefault("--games", "8"));
        int threads = Integer.parseInt(options.getOrDefault("--threads", "2"));
        long seed = Long.parseLong(options.getOrDefault("--seed", "20261012"));
        ComputerPlayerMCTS2.SHOW_THREAD_INFO = false;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<Map<String,Object>>> futures = new ArrayList<>();
        for (int i=0;i<games;i++) { final int index=i; futures.add(pool.submit(() -> play(deck, options, seed, index))); }
        pool.shutdown();
        List<Map<String,Object>> results = new ArrayList<>();
        try { for (Future<Map<String,Object>> future : futures) results.add(future.get(180, TimeUnit.SECONDS)); }
        finally { pool.shutdownNow(); }
        Map<String,Object> report = new LinkedHashMap<>(); report.put("options", options); report.put("games", results);
        Path output = Paths.get(options.get("--output")); if (output.getParent()!=null) Files.createDirectories(output.getParent());
        Files.write(output, new GsonBuilder().setPrettyPrinting().create().toJson(report).getBytes(StandardCharsets.UTF_8));
        System.out.println("Arena completed: " + games + " games; " + output);
    }

    private static Map<String,Object> play(DeckCardLists list, Map<String,String> options, long seed, int index) {
        Map<String,Object> result = new LinkedHashMap<>();
        result.put("index",index); result.put("seed",seed+index/2); result.put("learner_starts", index%2==0);
        long started=System.nanoTime(); PolicyPlayer a=null; Player b=null; Game game=null;
        boolean terminal=false;
        try {
            game = new TwoPlayerDuel(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE,
                MulliganType.GAME_DEFAULT.getMulligan(0),60,20,7);
            Match match = new TwoPlayerMatch(new MatchOptions("policy arena","policy",false));
            a = new PolicyPlayer("Learner",RangeOfInfluence.ONE,"http://127.0.0.1:"+options.getOrDefault("--port-a","50152"));
            String opponent=options.getOrDefault("--opponent","policy");
            if (opponent.equals("policy")) b=new PolicyPlayer("Opponent",RangeOfInfluence.ONE,
                "http://127.0.0.1:"+options.getOrDefault("--port-b","50153"));
            else if (opponent.equals("mcts-public") || opponent.equals("mcts-raw")) {
                ComputerPlayerMCTS2 search = opponent.equals("mcts-public") ? new PublicMCTS("Opponent")
                    : new ComputerPlayerMCTS2("Opponent",RangeOfInfluence.ONE,6);
                search.offlineMode=true; search.autoTap=true; search.allowMulligans=true;
                search.noPolicyPriority=true; search.noPolicyTarget=true; search.noPolicyUse=true; search.noPolicyOpponent=true;
                search.noNoise=true; search.searchBudget=Integer.parseInt(options.getOrDefault("--budget","300"));
                search.searchTimeout=Double.parseDouble(options.getOrDefault("--timeout","1.0"));
                b=search;
            } else throw new IllegalArgumentException("unsupported opponent " + opponent);
            load(game,match,a,Deck.load(list,false,false,new HashMap<>()));
            load(game,match,b,Deck.load(list,false,false,new HashMap<>()));
            a.initializeKnownDeck(game,b.getId());
            if (b instanceof PolicyPlayer) ((PolicyPlayer)b).initializeKnownDeck(game,a.getId());
            if (b instanceof ComputerPlayerMCTS2) {
                StateEncoder encoder=new StateEncoder(); encoder.setAgent(b.getId()); encoder.setOpponent(a.getId());
                encoder.perfectInfo=false; ((ComputerPlayerMCTS2)b).stateEncoder=encoder;
            }
            // Constructors in the original MCTS set a global/default RNG seed;
            // seed the actual game only after constructing both players.
            RandomUtil.setSeed(seed+index/2);
            GameOptions settings=new GameOptions(); settings.testMode=true; settings.stopOnTurn=150; settings.stopAtStep=PhaseStep.END_TURN;
            game.setGameOptions(settings); game.setStartingPlayerId(index%2==0?a.getId():b.getId());
            game.start(null); terminal=a.hasWon()||b.hasWon()||(a.hasLost()&&b.hasLost());
            int score=(a.hasLost()&&b.hasLost())?0:a.hasWon()?1:b.hasWon()?-1:0;
            result.put("terminal",terminal); result.put("result",score); result.put("turns",game.getTurnNum());
        } catch (Exception error) { result.put("terminal",false); result.put("error",error.toString()); }
        finally {
            try { if (a!=null&&game!=null) a.finish(game,terminal,false);
                  if (b instanceof PolicyPlayer&&game!=null) ((PolicyPlayer)b).finish(game,terminal,false); }
            catch(Exception error) { result.put("finish_error",error.toString()); }
            result.put("seconds",(System.nanoTime()-started)/1e9);
        }
        return result;
    }
    private static void load(Game game,Match match,Player player,Deck deck) {
        player.setTestMode(true); game.loadCards(deck.getCards(),player.getId()); game.loadCards(deck.getSideboard(),player.getId());
        game.addPlayer(player,deck); match.addPlayer(player,deck);
    }
}
