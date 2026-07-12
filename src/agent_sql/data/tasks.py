"""Task dataclass and the train/val split.

Tasks live in data/tasks.json (committed — reproducibility depends on it).
The gold SQL is used only by the reward function; the agent never sees it.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    """One evaluation task: an NL question with verified gold SQL."""

    id: str
    question: str
    gold_sql: str
    split: str  # "train" | "val"


def save_tasks(tasks: list[Task], path: str | Path) -> None:
    """Write tasks as a JSON list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(t) for t in tasks], indent=2), encoding="utf-8"
    )


def load_tasks(path: str | Path, split: str | None = None) -> list[Task]:
    """Load tasks, optionally filtered to one split."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = [Task(**t) for t in raw]
    if split is not None:
        tasks = [t for t in tasks if t.split == split]
    return tasks
