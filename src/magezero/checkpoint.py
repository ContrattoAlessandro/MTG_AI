"""Checkpoints: explicit architecture and catalog metadata, atomic writes, portable devices."""

import os
import uuid
from pathlib import Path

import torch

from .catalog import CardCatalog
from .model import EntityBeliefPolicy, ModelConfig, PrivilegedCritic

# 4: per-decision event pooling, final entity LayerNorm, pooled entities in the
# actor context, phase embedding, mean relation messages.
FORMAT_VERSION = 4


def architecture_name(config: ModelConfig) -> str:
    return (
        "entity_statistical_roles_recurrent_policy"
        if config.belief_mode == "statistical_roles"
        else "entity_belief_recurrent_policy"
    )


def resolve_device(name: str = "auto") -> torch.device:
    """`auto` selects CUDA when available; an explicit CUDA request without CUDA is an error."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _atomic_save(data: dict, path: Path) -> None:
    """Write through a temporary file, so a crash never leaves a partial checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    path, model, critic, catalog, *, optimizer=None, updates=0, policy_version=None, metrics=None
) -> str:
    """Save the actor, critic, catalog and optional optimizer state; return the policy version.

    The policy version identifies these exact weights: trajectories are tagged
    with it, and the learner rejects data collected by any other version.
    """
    version = policy_version or uuid.uuid4().hex
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "architecture": architecture_name(model.config),
        "model_config": model.config.to_dict(),
        "catalog": catalog.to_dict(),
        "policy_version": version,
        "updates": updates,
        "model_state_dict": model.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics or {},
    }
    _atomic_save(checkpoint, Path(path))
    return version


def load_checkpoint(path, device="cpu"):
    """Load and validate a checkpoint; return (model, critic, catalog, raw checkpoint dict).

    Both networks are returned in eval mode on `device`.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    version = checkpoint.get("format_version")
    if version in (2, 3):
        raise ValueError(
            "checkpoint predates the format-4 architecture (event pooling, final entity norm); "
            "its weights do not fit the current model: create a new one with mz init"
        )
    if version != FORMAT_VERSION:
        raise ValueError("expected a v4 entity-policy checkpoint; legacy MCTS weights require retraining")
    config = ModelConfig(**checkpoint["model_config"])
    if checkpoint.get("architecture") != architecture_name(config):
        raise ValueError("checkpoint belief mode disagrees with its architecture name")
    catalog = CardCatalog.from_dict(checkpoint["catalog"])
    if config.catalog_digest != catalog.digest or config.num_cards != catalog.num_cards:
        raise ValueError("checkpoint card catalog metadata is inconsistent")
    target = resolve_device(str(device))
    roles = catalog.role_tensor() if config.belief_mode == "statistical_roles" else None
    model = EntityBeliefPolicy(config, catalog.tensor(), roles).to(target)
    critic = PrivilegedCritic(config).to(target)
    model.load_state_dict(checkpoint["model_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    model.eval()
    critic.eval()
    return model, critic, catalog, checkpoint


def read_policy_version(path) -> str:
    """The policy version of a (validated) checkpoint."""
    return load_checkpoint(path)[3]["policy_version"]


def strip_optimizer(source, destination) -> Path:
    """Copy a checkpoint without optimizer state (about 1/3 the size), for opponents and play."""
    data = torch.load(source, map_location="cpu", weights_only=True)
    data["optimizer_state_dict"] = None
    destination = Path(destination)
    _atomic_save(data, destination)
    return destination
