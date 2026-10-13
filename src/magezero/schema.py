"""Versioned, player-relative observations and their tensor collation.

An observation is what one player may see at one decision: visible entities and
their relations, public global features, the events since that player's previous
decision, the legal candidate actions, and the publicly known opponent deck with
count bounds. Privileged training labels (the true opponent hand) are validated
separately and never enter an observation.

Opaque entity and action IDs are references only; they never become model
features. The engine derives observations from its own visibility rules; see
docs/policy-protocol.md for the field layout.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

SCHEMA_VERSION = 2
ENTITY_FEATURES = 48
GLOBAL_FEATURES = 24
ACTION_FEATURES = 16
EVENT_FEATURES = 8
CARD_FEATURES = 32
ACTION_TYPES = (
    "pass",
    "cast",
    "activate",
    "target",
    "use",
    "mode",
    "attack",
    "block",
    "order",
    "amount",
    "mulligan",
    "mana",
    "discard",
    "choose",
)
EVENT_TYPES = (
    "draw",
    "reveal",
    "zone_change",
    "cast",
    "activate",
    "resolve",
    "attack",
    "block",
    "damage",
    "life",
    "pass",
    "shuffle",
    "choice",
)


def _integer(value, label: str, minimum: int, maximum: int) -> int:
    """Return `value` if it is a (non-boolean) integer in [minimum, maximum]."""
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def numeric_vector(value, size: int, label: str) -> list[float]:
    """Return `value` as floats if it is a list of `size` finite numbers."""
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numeric features")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"{label} contains a non-finite or non-numeric feature")
    return [float(x) for x in value]


def validate_observation(obs: dict, config) -> dict:
    """Check an observation against schema v2 and the model's limits; return it unchanged.

    Raises ValueError for unknown fields (labels cannot enter the actor), a
    different card catalog, limits that would require truncation, dangling
    references, hidden information in hidden zones, non-incremental events, and
    public count bounds that cannot explain the opponent's hand size.
    """
    allowed = {
        "schema_version",
        "catalog_digest",
        "game_id",
        "player_id",
        "step",
        "entities",
        "relations",
        "globals",
        "events",
        "actions",
        "opponent_deck",
        "opponent_hand_size",
    }
    if not isinstance(obs, dict) or set(obs) - allowed:
        raise ValueError("unknown observation fields; training labels cannot enter the actor")
    required = allowed - {"relations"}
    if required - set(obs):
        raise ValueError(f"missing observation fields: {sorted(required - set(obs))}")
    if obs["schema_version"] != SCHEMA_VERSION:
        raise ValueError("incompatible observation schema; hashed-feature data is not convertible")
    if obs["catalog_digest"] != config.catalog_digest:
        raise ValueError("card catalog differs from the model checkpoint")
    for field in ("game_id", "player_id"):
        if not isinstance(obs[field], str) or not obs[field] or len(obs[field]) > 256:
            raise ValueError(f"{field} must be a nonempty string of at most 256 characters")
    _integer(obs["step"], "step", 0, 2**31 - 1)
    entities, actions = obs["entities"], obs["actions"]
    if not isinstance(entities, list) or not 1 <= len(entities) <= config.max_entities:
        raise ValueError("entity count is outside the configured limit; truncation is forbidden")
    if not isinstance(actions, list) or not 1 <= len(actions) <= config.max_actions:
        raise ValueError("legal action count is outside the configured limit")
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != {
            "id",
            "card_id",
            "kind",
            "zone",
            "owner",
            "controller",
            "visibility",
            "features",
        }:
            raise ValueError("entity fields do not match schema v2")
        _integer(entity["card_id"], "card_id", 0, config.num_cards - 1)
        _integer(entity["kind"], "kind", 0, 7)
        _integer(entity["zone"], "zone", 0, 7)
        _integer(entity["owner"], "owner", 0, 2)
        _integer(entity["controller"], "controller", 0, 2)
        features = numeric_vector(entity["features"], ENTITY_FEATURES, "entity.features")
        if entity["visibility"] not in ("public", "own", "hidden"):
            raise ValueError("invalid visibility")
        if entity["visibility"] == "own" and entity["owner"] != 0:
            raise ValueError("own private entities must belong to the observing player")
        # Slot 7 of a hidden-zone entity is its card count; nothing else may be set.
        hidden = entity["visibility"] == "hidden"
        if hidden and (entity["card_id"] != 0 or any(x != 0 for i, x in enumerate(features) if i != 7)):
            raise ValueError("hidden zones may expose only quantity, never identity or attributes")
    entity_ids = [entity["id"] for entity in entities]
    if any(not isinstance(x, str) or not x for x in entity_ids) or len(set(entity_ids)) != len(entity_ids):
        raise ValueError("entity references must be unique nonempty strings")
    for action in actions:
        if not isinstance(action, dict) or set(action) != {"id", "type", "source", "target", "card_id", "features"}:
            raise ValueError("action fields do not match schema v2")
        if action["type"] not in ACTION_TYPES:
            raise ValueError("unsupported action type")
        for ref in (action["source"], action["target"]):
            if ref is not None and ref not in entity_ids:
                raise ValueError("action references a missing entity")
        _integer(action["card_id"], "action.card_id", 0, config.num_cards - 1)
        numeric_vector(action["features"], ACTION_FEATURES, "action.features")
    action_ids = [a["id"] for a in actions]
    if any(not isinstance(x, str) or not x for x in action_ids) or len(set(action_ids)) != len(action_ids):
        raise ValueError("legal action ids must be unique nonempty strings")
    relations = obs.get("relations", [])
    if not isinstance(relations, list):
        raise ValueError("relations must be a list")
    for relation in relations:
        if not isinstance(relation, dict) or set(relation) != {"source", "target", "type"}:
            raise ValueError("invalid relation fields")
        if relation["source"] not in entity_ids or relation["target"] not in entity_ids:
            raise ValueError("relation references a missing entity")
        _integer(relation["type"], "relation.type", 0, 7)
    if len(obs.get("relations", [])) > config.max_relations:
        raise ValueError("too many relations")
    numeric_vector(obs["globals"], GLOBAL_FEATURES, "globals")
    events = obs["events"]
    if not isinstance(events, list) or len(events) > config.max_events:
        raise ValueError("too many incremental events; history must not be silently truncated")
    last = -1
    for event in events:
        if not isinstance(event, dict) or set(event) != {"seq", "type", "actor", "card_id", "features"}:
            raise ValueError("invalid event fields")
        seq = _integer(event["seq"], "event.seq", 0, 2**31 - 1)
        if seq <= last:
            raise ValueError("events must be ordered, unique, and incremental")
        last = seq
        if event["type"] not in EVENT_TYPES:
            raise ValueError("unsupported event type")
        _integer(event["actor"], "event.actor", 0, 2)
        _integer(event["card_id"], "event.card_id", 0, config.num_cards - 1)
        numeric_vector(event["features"], EVENT_FEATURES, "event.features")
        if event["type"] == "draw" and event["actor"] != 0 and event["card_id"] != 0:
            raise ValueError("opponent draw identities are private; emit a separate reveal event if public")
    deck = obs["opponent_deck"]
    if not isinstance(deck, list) or not 1 <= len(deck) <= config.max_deck_kinds:
        raise ValueError("invalid known opponent deck candidate set")
    seen = set()
    for card in deck:
        if not isinstance(card, dict) or set(card) != {"card_id", "lower", "upper"}:
            raise ValueError("deck entries must contain card_id and public count bounds")
        cid = _integer(card["card_id"], "deck.card_id", 1, config.num_cards - 1)
        if cid in seen:
            raise ValueError("duplicate deck card kind")
        seen.add(cid)
        lo = _integer(card["lower"], "lower", 0, config.max_hand_size)
        _integer(card["upper"], "upper", lo, 10000)
    size = _integer(obs["opponent_hand_size"], "opponent_hand_size", 0, config.max_hand_size)
    if not sum(x["lower"] for x in deck) <= size <= sum(x["upper"] for x in deck):
        raise ValueError("public card-count bounds cannot explain the opponent hand size")
    return obs


def validate_labels(labels: dict, obs: dict) -> list[int]:
    """Check the privileged labels of `obs`; return the true opponent hand counts.

    Counts follow the order of `obs["opponent_deck"]`, lie within its public
    bounds, and sum to the public hand size.
    """
    if not isinstance(labels, dict) or set(labels) != {"opponent_hand_counts"}:
        raise ValueError("training labels must contain opponent_hand_counts separately from observation")
    counts = labels["opponent_hand_counts"]
    if not isinstance(counts, list) or len(counts) != len(obs["opponent_deck"]):
        raise ValueError("label count ordering must match opponent_deck")
    for count, card in zip(counts, obs["opponent_deck"]):
        _integer(count, "opponent_hand_count", card["lower"], card["upper"])
    if sum(counts) != obs["opponent_hand_size"]:
        raise ValueError("training labels disagree with the observed hand size")
    return counts


@dataclass
class ObservationBatch:
    """Padded tensors for B observations; masks mark the real (unpadded) items.

    Shapes: entities [B, N], relations [B, R, 3] (source, target, type),
    globals [B, GLOBAL_FEATURES], events [B, E], actions [B, A] with entity
    pointers (-1 for none), opponent deck [B, K] with public count bounds, and
    the opponent hand size [B].
    """

    entity_cards: Tensor
    entity_categories: Tensor
    entity_features: Tensor
    entity_mask: Tensor
    relations: Tensor
    relation_mask: Tensor
    globals: Tensor
    event_cards: Tensor
    event_categories: Tensor
    event_features: Tensor
    event_mask: Tensor
    action_cards: Tensor
    action_types: Tensor
    action_sources: Tensor
    action_targets: Tensor
    action_features: Tensor
    action_mask: Tensor
    deck_cards: Tensor
    deck_lower: Tensor
    deck_upper: Tensor
    deck_mask: Tensor
    hand_size: Tensor

    def to(self, device) -> "ObservationBatch":
        """Move every tensor to `device`."""
        return ObservationBatch(**{key: value.to(device) for key, value in vars(self).items()})


def pad_cat(tensors: list[Tensor], fill: float = 0) -> Tensor:
    """Concatenate along dim 0, padding every other dim to the largest size."""
    shape = [max(t.shape[i] for t in tensors) for i in range(1, tensors[0].dim())]
    padded = []
    for tensor in tensors:
        if list(tensor.shape[1:]) != shape:
            out = tensor.new_full((tensor.shape[0], *shape), fill)
            out[(slice(None), *(slice(0, s) for s in tensor.shape[1:]))] = tensor
            tensor = out
        padded.append(tensor)
    return torch.cat(padded)


def concat_batches(batches: list[ObservationBatch]) -> ObservationBatch:
    """Join separately collated batches; the result equals collating all rows at once."""
    if len(batches) == 1:
        return batches[0]
    return ObservationBatch(
        **{
            key: pad_cat([vars(b)[key] for b in batches], -1 if key in ("action_sources", "action_targets") else 0)
            for key in vars(batches[0])
        }
    )


def _padded(rows: list[list], width: int, fill, dtype) -> Tensor:
    # One tensor construction per field; per-element tensors dominated runtime.
    return torch.tensor([row + [fill] * (width - len(row)) for row in rows], dtype=dtype)


def collate_observations(observations: list[dict], config, *, validate: bool = True) -> ObservationBatch:
    """Pad and stack observations into one batch.

    Pass validate=False only for observations this process has already validated.
    """
    if validate:
        for obs in observations:
            validate_observation(obs, config)
    if not observations:
        raise ValueError("empty observation batch")
    n = max(len(o["entities"]) for o in observations)
    r = max(1, max(len(o.get("relations", [])) for o in observations))
    e = max(1, max(len(o["events"]) for o in observations))
    a = max(len(o["actions"]) for o in observations)
    k = max(len(o["opponent_deck"]) for o in observations)
    long, real = torch.long, torch.float32
    refs = [{item["id"]: i for i, item in enumerate(o["entities"])} for o in observations]
    entities = [o["entities"] for o in observations]
    relations = [o.get("relations", []) for o in observations]
    events = [o["events"] for o in observations]
    actions = [o["actions"] for o in observations]
    decks = [o["opponent_deck"] for o in observations]

    def mask(groups, width):
        return _padded([[True] * len(g) for g in groups], width, False, torch.bool)

    return ObservationBatch(
        entity_cards=_padded([[x["card_id"] for x in g] for g in entities], n, 0, long),
        entity_categories=_padded(
            [[[x["kind"], x["zone"], x["owner"], x["controller"]] for x in g] for g in entities], n, [0, 0, 0, 0], long
        ),
        entity_features=_padded([[x["features"] for x in g] for g in entities], n, [0.0] * ENTITY_FEATURES, real),
        entity_mask=mask(entities, n),
        relations=_padded(
            [[[m[x["source"]], m[x["target"]], x["type"]] for x in g] for g, m in zip(relations, refs)],
            r,
            [0, 0, 0],
            long,
        ),
        relation_mask=mask(relations, r),
        globals=torch.tensor([o["globals"] for o in observations], dtype=real),
        event_cards=_padded([[x["card_id"] for x in g] for g in events], e, 0, long),
        event_categories=_padded(
            [[[EVENT_TYPES.index(x["type"]), x["actor"]] for x in g] for g in events], e, [0, 0], long
        ),
        event_features=_padded([[x["features"] for x in g] for g in events], e, [0.0] * EVENT_FEATURES, real),
        event_mask=mask(events, e),
        action_cards=_padded([[x["card_id"] for x in g] for g in actions], a, 0, long),
        action_types=_padded([[ACTION_TYPES.index(x["type"]) for x in g] for g in actions], a, 0, long),
        action_sources=_padded([[m.get(x["source"], -1) for x in g] for g, m in zip(actions, refs)], a, -1, long),
        action_targets=_padded([[m.get(x["target"], -1) for x in g] for g, m in zip(actions, refs)], a, -1, long),
        action_features=_padded([[x["features"] for x in g] for g in actions], a, [0.0] * ACTION_FEATURES, real),
        action_mask=mask(actions, a),
        deck_cards=_padded([[x["card_id"] for x in g] for g in decks], k, 0, long),
        deck_lower=_padded([[x["lower"] for x in g] for g in decks], k, 0, long),
        deck_upper=_padded([[x["upper"] for x in g] for g in decks], k, 0, long),
        deck_mask=mask(decks, k),
        hand_size=torch.tensor([o["opponent_hand_size"] for o in observations], dtype=long),
    )
