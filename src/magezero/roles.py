"""The 20 overlapping card roles (capabilities) used by the statistical hand prior.

Role membership is reviewed catalog metadata (configs/card-roles.json). A card
may have several roles, and roles are not mutually exclusive clusters.
"""

ROLE_NAMES = (
    "land_access", "mana_acceleration", "board_presence", "repeatable_engine",
    "card_draw", "card_selection", "tutor", "counterspell", "spot_removal",
    "sweeper", "tempo_interaction", "direct_reach", "combat_modifier",
    "protection", "hand_disruption", "recursion_cheating", "graveyard_hate",
    "persistent_restriction", "survival", "deck_synergy",
)


def validate_roles(roles: list) -> list:
    """Return `roles` if it is a list of distinct, supported role names."""
    if (not isinstance(roles, list) or any(not isinstance(role, str) or role not in ROLE_NAMES for role in roles)
            or len(roles) != len(set(roles))):
        raise ValueError("roles must be a unique list of supported role names")
    return roles
