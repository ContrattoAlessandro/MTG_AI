"""The immutable card catalog shared by the engine and the model.

Card IDs are contiguous and start at 1 (0 means unknown, hidden or padding).
Each card has a stable key, a display name, 32 static semantic features and,
in format 2, its reviewed overlapping roles (see roles.py). The SHA-256 digest
of the canonical JSON identifies the catalog in checkpoints and observations.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .roles import ROLE_NAMES, validate_roles
from .schema import CARD_FEATURES, numeric_vector


@dataclass(frozen=True)
class CardCatalog:
    cards: tuple[dict, ...]
    digest: str
    format_version: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "CardCatalog":
        """Validate a catalog in format 1 (features only) or 2 (features and roles)."""
        if (
            not isinstance(data, dict)
            or isinstance(data.get("format_version"), bool)
            or data.get("format_version") not in (1, 2)
        ):
            raise ValueError("expected card catalog format_version 1 or 2")
        version = data["format_version"]
        fields = {"format_version", "cards"} | ({"role_names"} if version == 2 else set())
        if set(data) != fields or (version == 2 and data["role_names"] != list(ROLE_NAMES)):
            raise ValueError("catalog role vocabulary/order is incompatible")
        cards = data["cards"]
        if not isinstance(cards, list) or not cards:
            raise ValueError("catalog must contain at least one card")
        keys = set()
        for i, card in enumerate(cards, 1):
            card_fields = {"id", "key", "name", "features"} | ({"roles"} if version == 2 else set())
            if (
                not isinstance(card, dict)
                or set(card) != card_fields
                or isinstance(card["id"], bool)
                or card["id"] != i
            ):
                raise ValueError("catalog IDs must be contiguous, start at 1, and keep their ordering")
            if not isinstance(card["key"], str) or not card["key"] or card["key"] in keys:
                raise ValueError("card keys must be unique stable identities")
            if not isinstance(card["name"], str) or not card["name"]:
                raise ValueError("card names must be nonempty strings")
            keys.add(card["key"])
            numeric_vector(card["features"], CARD_FEATURES, "card.features")
            if version == 2:
                validate_roles(card["roles"])
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return cls(tuple(json.loads(canonical)["cards"]), hashlib.sha256(canonical.encode()).hexdigest(), version)

    @classmethod
    def load(cls, path) -> "CardCatalog":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        result = {"format_version": self.format_version, "cards": list(self.cards)}
        if self.format_version == 2:
            result["role_names"] = list(ROLE_NAMES)
        return json.loads(json.dumps(result))

    def tensor(self) -> torch.Tensor:
        """Static card features, [num_cards, CARD_FEATURES]; row 0 is the zero padding card."""
        return torch.tensor([[0.0] * CARD_FEATURES] + [card["features"] for card in self.cards])

    def role_tensor(self) -> torch.Tensor:
        """Binary role membership, [num_cards, len(ROLE_NAMES)]; row 0 is empty."""
        if self.format_version != 2:
            raise ValueError("statistical_roles requires a v2 role catalog; run scripts/annotate_policy_catalog.py")
        return torch.tensor(
            [[0.0] * len(ROLE_NAMES)] + [[float(role in card["roles"]) for role in ROLE_NAMES] for card in self.cards]
        )

    @property
    def num_cards(self) -> int:
        """Embedding rows: every card plus ID 0 (unknown, hidden or padding)."""
        return len(self.cards) + 1
