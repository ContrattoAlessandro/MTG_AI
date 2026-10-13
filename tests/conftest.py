"""Shared fixtures. The package and scripts are importable through `pythonpath` in
pyproject.toml, so the tests also run without an installed package."""

import pytest
import torch

from magezero.catalog import CardCatalog
from magezero.model import EntityBeliefPolicy, ModelConfig, PrivilegedCritic


@pytest.fixture
def components():
    """A small learned-belief actor, its critic and a three-card catalog."""
    torch.manual_seed(2)
    catalog = CardCatalog.from_dict(
        {
            "format_version": 1,
            "cards": [{"id": i, "key": f"card-{i}", "name": f"Card {i}", "features": [0.0] * 32} for i in range(1, 4)],
        }
    )
    config = ModelConfig(
        num_cards=catalog.num_cards,
        catalog_digest=catalog.digest,
        width=32,
        entity_layers=1,
        max_hand_size=6,
        belief_mode="learned",
    )
    return EntityBeliefPolicy(config, catalog.tensor()).eval(), PrivilegedCritic(config), catalog
