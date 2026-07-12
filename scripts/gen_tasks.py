"""Generate (or --verify) the task set. Needs OPENAI_API_KEY in .env."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_sql.config import DB_PATH, N_TASKS, N_TRAIN  # noqa: E402
from agent_sql.data.generate_tasks import generate_tasks, verify_tasks  # noqa: E402
from agent_sql.data.tasks import load_tasks, save_tasks  # noqa: E402

DEFAULT_OUT = "data/tasks.json"


def main() -> int:
    """Generate verified tasks, or re-verify an existing task file."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=N_TASKS)
    parser.add_argument(
        "--verify", action="store_true", help="re-verify an existing task file and exit"
    )
    args = parser.parse_args()

    if args.verify:
        tasks = load_tasks(args.out)
        failures = verify_tasks(tasks, args.db)
        counts = Counter(t.split for t in tasks)
        print(f"{len(tasks)} tasks ({counts['train']} train / {counts['val']} val)")
        if failures:
            for task_id, reason in failures:
                print(f"  FAIL {task_id}: {reason}")
            return 1
        print("all gold SQL verified OK")
        return 0

    tasks = generate_tasks(args.db, n_tasks=args.n, n_train=min(N_TRAIN, args.n))
    save_tasks(tasks, args.out)
    counts = Counter(t.split for t in tasks)
    print(f"\nwrote {len(tasks)} tasks to {args.out} "
          f"({counts['train']} train / {counts['val']} val)")
    for t in tasks[:3]:
        print(f"  {t.id} [{t.split}] {t.question}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
