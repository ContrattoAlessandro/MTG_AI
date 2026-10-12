"""Entity attention + incremental event memory + hand belief + legal-action actor.

The actor has no API for private training labels. A separate privileged critic
receives those labels only inside the learner, through detached actor context.
"""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .belief import HandBelief, constrained_hand_belief
from .roles import ROLE_NAMES
from .statistics import role_statistics, statistical_hand_belief
from .schema import (ACTION_FEATURES, ACTION_TYPES, CARD_FEATURES, ENTITY_FEATURES,
                     EVENT_FEATURES, EVENT_TYPES, GLOBAL_FEATURES, ObservationBatch)


@dataclass(frozen=True)
class ModelConfig:
    num_cards: int = 4096
    catalog_digest: str = ""
    width: int = 256
    entity_layers: int = 2
    attention_heads: int = 4
    max_hand_size: int = 64
    max_entities: int = 512
    max_actions: int = 2048
    max_events: int = 1024
    max_relations: int = 2048
    max_deck_kinds: int = 128
    belief_mode: str = "statistical_roles"
    # Train a hand-count predictor from memory as an auxiliary loss only; its
    # output is never an actor input. Applies to statistical_roles mode.
    auxiliary_belief: bool = False

    def __post_init__(self):
        if self.belief_mode not in ("learned", "statistical_roles"):
            raise ValueError("belief_mode must be learned or statistical_roles")
        if not isinstance(self.auxiliary_belief, bool):
            raise ValueError("auxiliary_belief must be a boolean")
        if self.auxiliary_belief and self.belief_mode == "learned":
            raise ValueError("learned belief mode already trains its belief head; auxiliary_belief is redundant")
        for key in ("width", "num_cards", "entity_layers", "attention_heads", "max_hand_size", "max_entities",
                    "max_actions", "max_events", "max_relations", "max_deck_kinds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.width < 16 or self.width % self.attention_heads:
            raise ValueError("model width must be >=16 and divisible by attention_heads")
        if not isinstance(self.catalog_digest, str) or not self.catalog_digest:
            raise ValueError("a fixed card catalog digest is required")

    def to_dict(self):
        return asdict(self)


@dataclass
class PolicyMemory:
    events: Tensor
    decisions: Tensor

    def detach(self):
        return PolicyMemory(self.events.detach(), self.decisions.detach())


@dataclass
class PolicyOutput:
    logits: Tensor
    value: Tensor
    belief: HandBelief
    memory: PolicyMemory
    context: Tensor
    role_presence: Tensor | None = None
    role_expected_counts: Tensor | None = None
    auxiliary_count_logits: Tensor | None = None

    def distribution(self) -> Categorical:
        return Categorical(logits=self.logits)


@dataclass
class _Encoded:
    """Memory-independent features; computed in one batch for many timesteps."""
    pooled: Tensor
    global_state: Tensor
    events: Tensor
    action: Tensor


def mlp(input_size: int, width: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_size, width), nn.SiLU(),
                         nn.Linear(width, output_size))


class EntityBeliefPolicy(nn.Module):
    def __init__(self, config: ModelConfig, card_features: Tensor | None = None,
                 card_roles: Tensor | None = None):
        super().__init__()
        self.config = config
        d = config.width
        if card_features is None:
            card_features = torch.zeros(config.num_cards, CARD_FEATURES)
        if card_features.shape != (config.num_cards, CARD_FEATURES):
            raise ValueError("card feature matrix does not match the model catalog")
        self.register_buffer("card_features", card_features.float())
        self.card_ids = nn.Embedding(config.num_cards, d, padding_idx=0)
        self.card_semantics = nn.Linear(CARD_FEATURES, d, bias=False)
        self.categories = nn.ModuleList([nn.Embedding(size, d) for size in (8, 8, 3, 3)])
        self.entity_numeric = mlp(ENTITY_FEATURES, d, d)
        self.relation_types = nn.Embedding(8, d)
        self.relation_in = nn.Linear(d, d, bias=False)
        self.relation_out = nn.Linear(d, d, bias=False)
        self.entity_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, config.attention_heads, 4 * d,
                                            dropout=0.0, batch_first=True,
                                            norm_first=True, activation="gelu")
        self.entity_attention = nn.TransformerEncoder(layer, config.entity_layers,
                                                       enable_nested_tensor=False)
        # Pooling query is always a real token; no all-padding attention rows.
        self.pool_token = nn.Parameter(torch.zeros(1, 1, d))
        self.global_encoder = mlp(GLOBAL_FEATURES, d, d)
        self.event_types = nn.Embedding(len(EVENT_TYPES), d)
        self.event_actors = nn.Embedding(3, d)
        self.event_numeric = mlp(EVENT_FEATURES, d, d)
        self.event_norm = nn.LayerNorm(d)
        self.event_memory = nn.GRU(d, d, batch_first=True)
        # A sequence GRU (not GRUCell) so the learner can run a whole trajectory
        # chunk in one call; older GRUCell checkpoints are remapped on load.
        self.decision_memory = nn.GRU(3 * d, d, batch_first=True)
        self.register_load_state_dict_pre_hook(_upgrade_decision_memory)
        if config.belief_mode == "learned":
            self.belief_head = mlp(2 * d + 3, d, config.max_hand_size + 1)
        else:
            if (card_roles is None or card_roles.shape != (config.num_cards, len(ROLE_NAMES))
                    or not torch.all((card_roles == 0) | (card_roles == 1)) or card_roles[0].any()):
                raise ValueError("statistical_roles requires a binary catalog role matrix with empty padding")
            self.register_buffer("card_roles", card_roles.float())
            # One scalar per role: P(holding at least one). No role softmax.
            self.role_encoder = nn.Sequential(nn.Linear(len(ROLE_NAMES), d), nn.SiLU())
            if config.auxiliary_belief:
                self.auxiliary_belief_head = mlp(2 * d + 3, d, config.max_hand_size + 1)
        self.belief_encoder = mlp(d + 3, d, d)
        self.context_encoder = mlp(3 * d, d, d)
        self.action_types = nn.Embedding(len(ACTION_TYPES), d)
        self.action_numeric = mlp(ACTION_FEATURES, d, d)
        self.action_encoder = mlp(4 * d, d, d)
        self.action_scorer = mlp(3 * d, d, 1)
        self.value_head = mlp(d, d, 1)

    def initial_memory(self, batch_size: int, device=None) -> PolicyMemory:
        zero = self.pool_token.new_zeros(batch_size, self.config.width, device=device)
        return PolicyMemory(zero, zero.clone())

    def embed_card(self, ids: Tensor) -> Tensor:
        return self.card_ids(ids) + self.card_semantics(self.card_features[ids])

    def forward(self, batch: ObservationBatch, memory: PolicyMemory | None = None,
                reset: Tensor | None = None) -> PolicyOutput:
        """One decision for each of B independent sessions (live inference)."""
        b = batch.entity_cards.shape[0]
        if memory is None:
            memory = self.initial_memory(b, batch.entity_cards.device)
        if reset is not None:
            keep = (~reset.bool()).to(memory.events.dtype).unsqueeze(-1)
            memory = PolicyMemory(memory.events * keep, memory.decisions * keep)
        encoded = self._encode(batch)
        lengths = batch.event_mask.sum(-1)
        packed = pack_padded_sequence(encoded.events, lengths.clamp(min=1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, event_hidden = self.event_memory(packed, memory.events.unsqueeze(0))
        event_hidden = torch.where((lengths > 0).unsqueeze(-1), event_hidden[0], memory.events)
        step_input = torch.cat([encoded.pooled, encoded.global_state, event_hidden], -1)
        _, hidden = self.decision_memory(step_input.unsqueeze(1), memory.decisions.unsqueeze(0))
        return self._head(batch, encoded, hidden[0], PolicyMemory(event_hidden, hidden[0]))

    def forward_sequence(self, batch: ObservationBatch,
                         memory: PolicyMemory | None = None) -> PolicyOutput:
        """Consecutive decisions of ONE trajectory, batched over time.

        Row t of `batch` is step t; memory has batch size one. Identical to calling
        forward() step by step, but the entity encoder, belief and action scorer
        run once for all rows, and each GRU runs once over its whole sequence.
        """
        steps = batch.entity_cards.shape[0]
        if memory is None:
            memory = self.initial_memory(1, batch.entity_cards.device)
        if memory.events.shape[0] != 1:
            raise ValueError("forward_sequence carries the memory of exactly one trajectory")
        encoded = self._encode(batch)
        # Incremental events of consecutive steps form one continuous stream.
        # Row-major masking keeps time order, and each step reads the hidden state
        # after its own last event (or the carried state when it has none yet).
        stream = memory.events
        flat = encoded.events[batch.event_mask]
        if flat.shape[0]:
            outputs, _ = self.event_memory(flat.unsqueeze(0), memory.events.unsqueeze(0))
            stream = torch.cat([memory.events, outputs[0]], 0)
        event_hidden = stream[batch.event_mask.sum(-1).cumsum(0)]
        step_input = torch.cat([encoded.pooled, encoded.global_state, event_hidden], -1)
        hidden, _ = self.decision_memory(step_input.unsqueeze(0), memory.decisions.unsqueeze(0))
        hidden = hidden[0]
        if hidden.shape[0] != steps:
            raise ValueError("sequence recurrence lost timesteps")
        return self._head(batch, encoded, hidden, PolicyMemory(event_hidden[-1:], hidden[-1:]))

    def forward_episodes(self, batch: ObservationBatch, lengths: list[int],
                         memory: PolicyMemory | None = None,
                         event_counts: Tensor | None = None) -> PolicyOutput:
        """Consecutive decisions of SEVERAL trajectories in one pass (the learner path).

        Rows are episode-major: the first lengths[0] rows are consecutive steps of
        trajectory 0, and so on. Memory has one row per trajectory and is returned
        the same way. Equivalent to forward_sequence on each trajectory separately:
        every GRU runs once over padded, packed per-trajectory sequences.

        event_counts optionally gives each row's event count as a CPU tensor (the
        learner keeps it from collation), so packing needs no device-to-host sync.
        """
        n, rows, d = len(lengths), batch.entity_cards.shape[0], self.config.width
        device = batch.entity_cards.device
        if n < 1 or min(lengths) < 1 or sum(lengths) != rows:
            raise ValueError("trajectory lengths must be positive and cover every row")
        if memory is None:
            memory = self.initial_memory(n, device)
        if memory.events.shape[0] != n:
            raise ValueError("memory needs one row per trajectory")
        encoded = self._encode(batch)
        sizes_cpu = torch.tensor(lengths)
        sizes = sizes_cpu.to(device)
        episode = torch.repeat_interleave(torch.arange(n, device=device), sizes, output_size=rows)
        position = torch.arange(rows, device=device) - (sizes.cumsum(0) - sizes)[episode]
        # Each trajectory's incremental events form one continuous stream; a step
        # reads the hidden state after its own last event (index 0 = carried state).
        counts_cpu = (batch.event_mask.sum(-1) if event_counts is None else event_counts).cpu().long()
        if counts_cpu.shape != (rows,):
            raise ValueError("event_counts needs one count per row")
        totals_cpu = torch.stack([part.sum() for part in counts_cpu.split(lengths)])
        counts, totals = counts_cpu.to(device), totals_cpu.to(device)
        starts = totals.cumsum(0) - totals
        within = counts.cumsum(0) - starts[episode]
        longest, events_total = int(totals_cpu.max()), int(totals_cpu.sum())
        if longest:
            flat = encoded.events[batch.event_mask]
            owner = torch.repeat_interleave(episode, counts, output_size=events_total)
            padded = flat.new_zeros(n, longest, d)
            padded[owner, torch.arange(events_total, device=device) - starts[owner]] = flat
            # A trajectory without events gets one ignored zero input; its rows
            # and returned memory all read index 0, the carried state.
            packed = pack_padded_sequence(padded, totals_cpu.clamp(min=1), batch_first=True,
                                          enforce_sorted=False)
            outputs, _ = self.event_memory(packed, memory.events.unsqueeze(0))
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=longest)
            stream = torch.cat([memory.events.unsqueeze(1), outputs], 1)
        else:
            stream = memory.events.unsqueeze(1)
        event_hidden = stream[episode, within]
        final_events = stream[torch.arange(n, device=device), totals]
        step_input = torch.cat([encoded.pooled, encoded.global_state, event_hidden], -1)
        padded = step_input.new_zeros(n, max(lengths), step_input.shape[-1])
        padded[episode, position] = step_input
        packed = pack_padded_sequence(padded, sizes_cpu, batch_first=True, enforce_sorted=False)
        outputs, final = self.decision_memory(packed, memory.decisions.unsqueeze(0))
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=max(lengths))
        hidden = outputs[episode, position]
        return self._head(batch, encoded, hidden, PolicyMemory(final_events, final[0]))

    def _encode(self, batch: ObservationBatch) -> _Encoded:
        b = batch.entity_cards.shape[0]
        d = self.config.width
        entity = self.embed_card(batch.entity_cards) + self.entity_numeric(batch.entity_features)
        for column, embedding in enumerate(self.categories):
            entity = entity + embedding(batch.entity_categories[..., column])
        # Directed relation messages preserve attachment, combat, and target links.
        source, target, kind = batch.relations.unbind(-1)
        gather_source = entity.gather(1, source.unsqueeze(-1).expand(-1, -1, d))
        gather_target = entity.gather(1, target.unsqueeze(-1).expand(-1, -1, d))
        rel = self.relation_types(kind)
        valid = batch.relation_mask.unsqueeze(-1)
        incoming = (self.relation_in(gather_source) + rel) * valid
        outgoing = (self.relation_out(gather_target) + rel) * valid
        messages = torch.zeros_like(entity)
        messages.scatter_add_(1, target.unsqueeze(-1).expand(-1, -1, d), incoming)
        messages.scatter_add_(1, source.unsqueeze(-1).expand(-1, -1, d), outgoing)
        entity = self.entity_norm(entity + messages)
        tokens = torch.cat([self.pool_token.expand(b, -1, -1), entity], 1)
        valid_tokens = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=entity.device),
                                  batch.entity_mask], 1)
        tokens = self.entity_attention(tokens, src_key_padding_mask=~valid_tokens)
        pooled, entity = tokens[:, 0], tokens[:, 1:]
        event = (self.embed_card(batch.event_cards) + self.event_types(batch.event_categories[..., 0])
                 + self.event_actors(batch.event_categories[..., 1]) + self.event_numeric(batch.event_features))

        def pointer(refs: Tensor) -> Tensor:
            values = entity.gather(1, refs.clamp(min=0).unsqueeze(-1).expand(-1, -1, d))
            return values * (refs >= 0).unsqueeze(-1)
        action = self.embed_card(batch.action_cards) + self.action_types(batch.action_types)
        action = action + self.action_numeric(batch.action_features)
        source_vectors, target_vectors = pointer(batch.action_sources), pointer(batch.action_targets)
        action = self.action_encoder(torch.cat([action, source_vectors, target_vectors,
                                                action * (source_vectors + target_vectors)], -1))
        return _Encoded(pooled, self.global_encoder(batch.globals), self.event_norm(event), action)

    def _head(self, batch: ObservationBatch, encoded: _Encoded, hidden: Tensor,
              memory: PolicyMemory) -> PolicyOutput:
        global_state = encoded.global_state
        cards = self.embed_card(batch.deck_cards)
        role_presence = role_expected = auxiliary_logits = None
        k = cards.shape[1]
        bounds = torch.stack([batch.deck_lower.float() / self.config.max_hand_size,
                              batch.deck_upper.float().clamp(max=self.config.max_hand_size) / self.config.max_hand_size,
                              batch.hand_size.float()[:, None].expand(-1, k) / self.config.max_hand_size], -1)
        count_input = torch.cat([cards, hidden[:, None].expand(-1, k, -1), bounds], -1)
        if self.config.belief_mode == "learned":
            count_logits = self.belief_head(count_input)
            belief = constrained_hand_belief(count_logits, batch.deck_lower, batch.deck_upper, batch.hand_size)
        else:
            if self.config.auxiliary_belief:
                # Scores only; the learner evaluates the count DP. Nothing here
                # flows into context, logits or value.
                auxiliary_logits = self.auxiliary_belief_head(count_input)
            # Bounds were checked by validate_observation; skip per-pass host syncs.
            belief = statistical_hand_belief(batch.deck_lower, batch.deck_upper, batch.hand_size,
                                             self.config.max_hand_size + 1, validate=False)
            role_presence, role_expected = role_statistics(batch.deck_lower, batch.deck_upper,
                batch.hand_size, self.card_roles[batch.deck_cards], validate=False)
        probs = belief.probabilities
        entropy = -(probs * probs.clamp(min=1e-12).log()).sum(-1)
        summary_features = torch.cat([cards,
            (belief.expected_counts / self.config.max_hand_size).unsqueeze(-1),
            (1 - probs[..., 0]).unsqueeze(-1), entropy.unsqueeze(-1)], -1)
        summaries = self.belief_encoder(summary_features) * batch.deck_mask.unsqueeze(-1)
        belief_summary = summaries.sum(1) / batch.deck_mask.sum(1, keepdim=True).clamp(min=1)
        if role_presence is not None:
            belief_summary = belief_summary + self.role_encoder(role_presence)
        context = self.context_encoder(torch.cat([hidden, global_state, belief_summary], -1))
        action = encoded.action
        state = context[:, None].expand_as(action)
        logits = self.action_scorer(torch.cat([state, action, state * action], -1)).squeeze(-1)
        logits = logits.masked_fill(~batch.action_mask, float("-inf"))
        return PolicyOutput(logits, self.value_head(context).squeeze(-1), belief,
                            memory, context, role_presence, role_expected, auxiliary_logits)


def _upgrade_decision_memory(module, state_dict, prefix, *args):
    """Map pre-sequence GRUCell weights onto the equivalent one-layer GRU."""
    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        old = f"{prefix}decision_memory.{name}"
        if old in state_dict:
            state_dict[f"{old}_l0"] = state_dict.pop(old)


class PrivilegedCritic(nn.Module):
    """Training-only baseline. Its card embeddings are independent of the actor."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cards = nn.Embedding(config.num_cards, config.width, padding_idx=0)
        self.value = mlp(2 * config.width + 1, config.width, 1)

    def forward(self, context: Tensor, card_ids: Tensor, true_counts: Tensor) -> Tensor:
        # No privileged input or privileged gradients enter actor representations.
        summary = (self.cards(card_ids) * true_counts.unsqueeze(-1)).sum(1)
        size = true_counts.sum(1, keepdim=True)
        summary = summary / size.clamp(min=1)
        return self.value(torch.cat([context.detach(), summary, size / 64.0], -1)).squeeze(-1)
