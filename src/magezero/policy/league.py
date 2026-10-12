"""Small league for historical opponents; sampling is not a Nash guarantee."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random


@dataclass
class OpponentRecord:
    checkpoint: str
    wins: float = 0.0
    games: int = 0


class League:
    def __init__(self, opponents=None):
        self.opponents = opponents or []

    def add(self, checkpoint):
        path = str(Path(checkpoint).resolve())
        if not Path(path).is_file():
            raise ValueError("league snapshot checkpoint does not exist")
        if path not in {x.checkpoint for x in self.opponents}:
            self.opponents.append(OpponentRecord(path))

    def record(self, checkpoint, result):
        if result not in (-1, 0, 1):
            raise ValueError("league result must be from the learner's perspective")
        record = next(x for x in self.opponents if x.checkpoint == str(Path(checkpoint).resolve()))
        record.games += 1
        record.wins += (result + 1) / 2

    def sample(self, rng=None):
        if not self.opponents:
            raise ValueError("league is empty")
        rng = rng or random
        # Prioritized fictitious self-play: emphasize opponents we have not beaten;
        # mix in uniform sampling to retain diversity, smooth untested records.
        winrates = [(x.wins + 1) / (x.games + 2) for x in self.opponents]
        weights = [(1 - w)**2 for w in winrates]
        total = sum(weights)
        probabilities = [0.2 / len(weights) + 0.8 * w / total for w in weights]
        return rng.choices(self.opponents, weights=probabilities, k=1)[0].checkpoint

    def save(self, path):
        Path(path).write_text(json.dumps({"format_version": 1,
            "opponents": [asdict(x) for x in self.opponents]}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["format_version"] != 1:
            raise ValueError("incompatible league format")
        return cls([OpponentRecord(**x) for x in data["opponents"]])
