"""Attach reviewed, overlapping roles without changing engine card IDs.

python scripts/annotate_policy_catalog.py --catalog artifacts/real-card-catalog.json \
    --annotations configs/card-roles.json --output artifacts/statistical-catalog.json
Missing annotations fail explicitly: an absent label must not silently mean no capability.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from magezero.policy.catalog import CardCatalog
from magezero.policy.roles import ROLE_NAMES, validate_roles


def annotate_catalog(catalog, annotations):
    if not isinstance(annotations, dict) or set(annotations) != {"format_version", "cards"} or annotations["format_version"] != 1:
        raise ValueError("expected reviewed annotations format_version 1")
    entries = annotations["cards"]
    if not isinstance(entries, dict):
        raise ValueError("annotations.cards must map exact card names to reviewed entries")
    for name, entry in entries.items():
        if not isinstance(name, str) or not isinstance(entry, dict) or set(entry) != {"roles", "source", "notes"}:
            raise ValueError("each annotation requires roles, source, and notes")
        validate_roles(entry["roles"])
        if not all(isinstance(entry[key], str) and entry[key] for key in ("source", "notes")):
            raise ValueError("annotations must record their review source and notes")
    missing = [card["name"] for card in catalog.cards if card["name"] not in entries]
    if missing:
        raise ValueError("missing reviewed card roles: " + ", ".join(missing))
    data = {"format_version": 2, "role_names": list(ROLE_NAMES), "cards": []}
    for card in catalog.cards:
        data["cards"].append({**card, "roles": [role for role in ROLE_NAMES if role in entries[card["name"]]["roles"]]})
    return CardCatalog.from_dict(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--annotations", default="configs/card-roles.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("choose a fresh output path; catalog changes require a new checkpoint")
    catalog = annotate_catalog(CardCatalog.load(args.catalog), json.loads(Path(args.annotations).read_text(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Annotated {len(catalog.cards)} card kinds; digest={catalog.digest}")


if __name__ == "__main__":
    main()
