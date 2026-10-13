"""Potential-based reward shaping from the observing player's public state.

F(s, s') = gamma * phi(s') - phi(s), with phi = 0 after a terminal state, leaves the
optimal policy unchanged (Ng, Harada & Russell 1999): the shaped return of every
step differs from the true one only by -phi(s_t), a state-dependent baseline. With
gamma = lambda = 1 it therefore changes nothing but the value targets; it helps
when GAE bootstraps (lambda < 1), where it acts as a value initialization
(Wiewiora 2003) while the critic is still poor.

phi is computed only from the actor's observation (no private labels), so the same
potential is valid for both players and at any decision.
"""

import math

import torch

# Player entities: kind 0, zone 0. Battlefield objects: zone 1.
_LIFE, _POISON, _COMMANDER_DAMAGE = 11, 12, 15  # player entity slots (/40, /10, /21)
_POWER, _CREATURE, _LAND = 0, 21, 22  # object entity slots (/20, flags)
_HAND_SELF, _HAND_OPPONENT = 5, 6  # global slots (/60)


def _health(features) -> float:
    """The nearest loss condition: life, poison, or commander damage taken, in [0, 1]."""
    return min(features[_LIFE], 1.0 - features[_POISON], 1.0 - features[_COMMANDER_DAMAGE])


def potential(observation: dict) -> float:
    """Heuristic advantage of the observer, squashed to (-1, 1); 0 for an even position."""
    health = {0: 0.0, 1: 0.0}
    permanents = {0: 0.0, 1: 0.0}
    power = {0: 0.0, 1: 0.0}
    for entity in observation["entities"]:
        side = entity["controller"]
        if side not in (0, 1) or entity["visibility"] == "hidden":
            continue
        features = entity["features"]
        if entity["kind"] == 0 and entity["zone"] == 0:
            health[side] = _health(features)
        elif entity["zone"] == 1 and features[_LAND] == 0:
            permanents[side] += 1
            if features[_CREATURE]:
                power[side] += max(0.0, features[_POWER] * 20)
    hands = observation["globals"]
    x = (
        1.0 * (health[0] - health[1])
        + 0.25 * (hands[_HAND_SELF] - hands[_HAND_OPPONENT]) * 60 / 7
        + 0.25 * (permanents[0] - permanents[1]) / 10
        + 0.5 * (power[0] - power[1]) / 20
    )
    return math.tanh(x)


def shaped_rewards(episode: dict, coefficient: float, gamma: float = 1.0) -> torch.Tensor:
    """Rewards plus gamma * phi(s_{t+1}) - phi(s_t), with phi scaled by `coefficient`.

    After the last decision phi is 0 for a finished game; a truncated game has no
    next observation, so its last step reuses its own potential (no shaping term at
    gamma = 1), consistent with bootstrapping from that state's value.
    """
    steps = episode["steps"]
    rewards = torch.tensor([float(step["reward"]) for step in steps], dtype=torch.float64)
    if coefficient == 0:
        return rewards.float()
    phi = torch.tensor([coefficient * potential(step["observation"]) for step in steps], dtype=torch.float64)
    following = torch.cat([phi[1:], phi[-1:] if episode.get("truncated", False) else phi.new_zeros(1)])
    return (rewards + gamma * following - phi).float()
