"""Stable, overlapping capabilities; membership is reviewed catalog metadata."""

ROLE_NAMES = (
    "land_access", "mana_acceleration", "board_presence", "repeatable_engine",
    "card_draw", "card_selection", "tutor", "counterspell", "spot_removal",
    "sweeper", "tempo_interaction", "direct_reach", "combat_modifier",
    "protection", "hand_disruption", "recursion_cheating", "graveyard_hate",
    "persistent_restriction", "survival", "deck_synergy",
)


def validate_roles(roles):
    if (not isinstance(roles, list) or any(not isinstance(role, str) or role not in ROLE_NAMES for role in roles)
            or len(roles) != len(set(roles))):
        raise ValueError("roles must be a unique list of supported role names")
    return roles
