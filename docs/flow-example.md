# One actual MageZero decision, from observation to learning

This example comes from the first pilot training generation, observer decision **29**, during the precombat main phase of turn **4**. Both players have 20 life. The observer has seven hand cards; the opponent has five. There are fifteen visible/aggregate entities, three new events, no explicit relation edges, and five legal action candidates. This is recorded data, not invented probabilities.

The initial checkpoint and the checkpoint after the first generation are replayed on exactly the same observation history. The source episode and raw values are in `artifacts/pilot-2026-10-12/flow-onpolicy-before.json`, `flow-onpolicy-after.json`, and `flow-training.json`. After-update outputs are diagnostic replay values; they were not the probabilities used to collect this game.

## 1. XMage supplies the observer's information

The fifteen entities cover public players, aggregate hidden libraries, the observer's own hand, and visible battlefield objects. There are no opponent-hand card entities. Opaque UUIDs connect references but are not embedded as card identities.

Public globals include `[turn/100=.04, self_active=1, stack_size/16=0, self_life/40=.5, opponent_life/40=.5, self_hand/60=.1167, opponent_hand/60=.0833, ... phase_ordinal/32=.09375]`. Phase ordinal 3 is precombat main. Other global slots carry public mana pools.

The legal candidates are pass, cast Evolving Adaptive, or play one of three physical Forest cards. All three Forest actions have the same strategic meaning in this position, but refer to different hand objects. The adapter currently labels land plays as action type `activate`; their land semantics come from the card/entity features. The recorded action was `priority-4`, the third Forest candidate.

Events 98–100 describe the observer's previous choice, its pass, and the opponent's pass. Only these new events enter this call; the prior 29 decisions have already contributed to memory.

## 2. Build fifteen entity vectors

Each entity combines learned card identity, a projection of the catalog's 32 static card features, a projection of its 32 dynamic features, and embeddings for kind/zone/owner/controller. The result has shape **[1,15,256]**: one batch row, fifteen objects, 256 learned coordinates per object.

A Forest identity embedding distinguishes its card kind from Evolving Adaptive. Static features describe basic card semantics; dynamic features describe the particular instance's current state. Two instances can share card identity but have different tapping, zone, controller, damage, or counter features. Player/library entities use identity 0, with categories and numeric attributes providing their meaning.

Relations normally add messages between attached objects, targets, attackers and blockers. This observation has no edges, so relation messages are zero. We do not invent combat or attachment edges for this example.

The coordinates are learned numbers, not individually named concepts such as “aggression.” Their norms and first values in the trace demonstrate actual transformation, not human-readable reasoning.

## 3. Attention contextualizes the board

A learned pooling token is added, giving **[1,16,256]**. Two Transformer layers process the set. Each layer has four attention heads, 64 coordinates per head, and a 1,024-wide feed-forward sublayer.

Conceptually, a head computes `softmax(Q Kᵀ / sqrt(64)) V`. Objects can exchange information about the visible position. The output retains a contextual vector for each entity and a global vector from the pooling token.

The trace computes the actual first layer's pooling-query attention from its projection weights and inputs. Its initial pooling query starts uniformly here; it changes after learning. Those weights illustrate one operation and are not causal explanations for the final selected action. The network has another attention layer, recurrent memory, belief processing, and action scoring afterward.

## 4. Update two memories

The event GRU embeds event kind, actor, any visible card identity and eight event features, then updates its **[1,256]** hidden state from the three new events. An opponent's hidden draw would have identity 0.

The decision GRU cell receives the concatenation of the pooling vector, public global encoding, and event memory: **[1,768] → [1,256]**. It also receives the previous decision hidden state. Thus this call depends on observed history as well as the current snapshot.

During play these memories persist until this player's game finishes. During PPO they are reconstructed from the beginning at the evaluated weights; gradients detach every 32 decisions. Detaching gradients does not clear memory.

## 5. Infer the opponent's hand

The announced deck catalog contains ten card kinds. For each one, the head receives its 256-dimensional card embedding, the decision memory, and three public quantities: lower count bound, upper count bound, and hand size. This is **[1,10,515] → [1,10,65]**: scores for counts zero through 64.

Only feasible counts survive. The joint distribution is proportional to `exp(sum_i score_i(count_i))`, with public bounds and total hand size exactly five. Log-space dynamic programming computes its normalizer and per-card marginals. This performs arithmetic over counts; it does not play hypothetical games.

For Forest, the public range is 0–22. Before the update, expected count was **0.504** and probability of at least one was **33.0%**. After the first generation, those became **0.673** and **53.0%**. The true hand had **two Forests**, visible to the trainer only after the game. The example remains inaccurate despite aggregate improvements elsewhere.

Constraints alone do not give a physical shuffle prior. A uniform score for each count vector is different from drawing physical cards without replacement. The pilot evaluates a hypergeometric known-deck prior separately; a residual over that prior is a useful next model experiment.

For the actor, each card's expected count, presence probability and marginal entropy are combined with its embedding, producing **[1,10,256]**. These summaries are pooled and merged with decision memory and global context into **[1,256]**. Expected counts over all ten kinds sum to five.

The actor receives these predictions, not the trainer's actual hidden hand.

## 6. Score and execute one legal action

Each candidate combines card/action-type embeddings, numeric/lexical features, and contextual entity pointers. The three Forest source indices are 4, 5, and 10; Evolving Adaptive's is 8. Pass has source index −1, meaning no source pointer.

The candidate encoder takes the candidate vector, source vector, target vector, and an interaction term: **[1,5,1024] → [1,5,256]**. Each candidate is then scored with actor context and their elementwise interaction: **[1,5,768] → [1,5,1]**. Softmax normalizes the five logits.

| Candidate | Initial probability | After first generation |
|---|---:|---:|
| Pass | 19.92% | 25.46% |
| Cast Evolving Adaptive | 20.19% | 17.14% |
| Play Forest, instance 1 | 19.96% | 19.13% |
| Play Forest, instance 2 | 19.96% | 19.13% |
| Play Forest, instance 3; recorded choice | 19.96% | 19.14% |

The acting service sampled the third Forest using the initial distribution. XMage then performed the land play. The engine establishes legality and handles mana payment; the network does not search future states. The new probabilities prove the policy changed, but do not prove that preferring pass more often is better.

The public value head also returns a scalar. Initially it was **0.05785**. It is an unbounded return estimate, not a calibrated win probability and not a score used for search.

## 7. Learn after the outcome is known

This observer eventually lost, giving terminal reward **−1**. Earlier recorded rewards are zero. Actual opponent counts at every recorded decision become separate training labels. They do not become actor input fields.

The independent privileged critic embeds and pools the true hand, concatenates it with detached actor context and hand size, and predicts a value. Its initial value here was **0.08279**. Its own loss cannot backpropagate through detached actor context. Policy advantages and the belief-supervision loss still train the actor through their legitimate training paths.

The learner uses that critic along the full trajectory to calculate GAE. For this decision, return target was **0.08807**, raw advantage **0.00529**, and advantage after normalization across the generation **0.01096**. A positive local advantage in an eventually lost game is possible: it is an estimate relative to the critic, not a judgment that every action in a losing game was bad.

Here the initial selected-action probability was **0.199619** and the after-update diagnostic probability **0.191351**. Their ratio is **0.958579**, within PPO's `[0.8,1.2]` clipping range. The endpoint policy-loss contribution is `-min(r*A, clip(r)*A) = -0.010504`.

Other diagnostic contributions are privileged-value MSE **0.009729**, public-value MSE **0.007054**, belief NLL/card **0.716397**, and policy entropy **1.599859**. With coefficients `1, .5, .25, .1, -.01`, their weighted sum is **0.051765** before minibatch normalization. This is one sample evaluated at endpoint weights, not the recorded last optimizer loss.

Training combines many such samples. Every 32 decisions, backpropagation releases that chunk's graph and detaches recurrent states; gradients accumulate until a complete episode minibatch finishes. They are clipped, AdamW updates weights, and subsequent PPO passes rebuild memory at the new weights. The first generation contained eight player episodes, producing four optimizer steps over two epochs.

The chosen action's probability need not rise simply because this one advantage was positive: other samples, belief supervision, public value regression, entropy and shared representations all affect the same update. A mean PPO loss near zero also does not imply zero gradients; normalized advantages can cancel in the scalar mean while individual gradient contributions differ.

## Why this matters for the next experiment

With gamma 1 and lambda 0.97, the direct terminal-reward contribution shrinks strongly over long micro-decision gaps. After 300 decisions its lambda factor is only about 0.000108. An inaccurate early critic can therefore dominate targets, as the positive target in this eventually lost example illustrates. A lambda-1/terminal-return ablation should be measured before long training.

The current actor is a small recurrent entity model with a useful but imperfect belief head. This complete trace establishes which computations occurred and where learning signals enter. It does not establish that its policy is optimal or stronger than upstream MageZero.
