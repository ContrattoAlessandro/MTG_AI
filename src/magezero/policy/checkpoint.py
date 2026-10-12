"""Explicit architecture/catalog metadata, atomic writes, portable devices."""

import os
from pathlib import Path
import uuid

import torch

from .catalog import CardCatalog
from .model import EntityBeliefPolicy, ModelConfig, PrivilegedCritic

FORMAT_VERSION = 3


def architecture_name(config):
    return ("entity_statistical_roles_recurrent_policy" if config.belief_mode == "statistical_roles"
            else "entity_belief_recurrent_policy")


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def save_checkpoint(path, model, critic, catalog, *, optimizer=None, updates=0,
                    policy_version=None, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    version = policy_version or uuid.uuid4().hex
    checkpoint = {
        "format_version": FORMAT_VERSION if model.config.belief_mode == "statistical_roles" else 2,
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
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return version


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    version = checkpoint.get("format_version")
    expected = {2: "entity_belief_recurrent_policy", 3: "entity_statistical_roles_recurrent_policy"}
    if version not in expected or checkpoint.get("architecture") != expected[version]:
        raise ValueError("expected a v2/v3 entity-policy checkpoint; legacy MCTS weights require retraining")
    raw_config = dict(checkpoint["model_config"])
    raw_config.setdefault("belief_mode", "learned" if version == 2 else "statistical_roles")
    config = ModelConfig(**raw_config)
    if architecture_name(config) != expected[version]:
        raise ValueError("checkpoint belief mode disagrees with its architecture version")
    catalog = CardCatalog.from_dict(checkpoint["catalog"])
    if config.catalog_digest != catalog.digest or config.num_cards != catalog.num_cards:
        raise ValueError("checkpoint card catalog metadata is inconsistent")
    target = resolve_device(str(device))
    model = EntityBeliefPolicy(config, catalog.tensor(),
        catalog.role_tensor() if config.belief_mode == "statistical_roles" else None).to(target)
    critic = PrivilegedCritic(config).to(target)
    model.load_state_dict(checkpoint["model_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    model.eval()
    critic.eval()
    return model, critic, catalog, checkpoint
