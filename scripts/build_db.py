"""Build the synthetic e-commerce SQLite DB (deterministic seed)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_sql.config import DB_PATH  # noqa: E402
from agent_sql.db.build import build_db  # noqa: E402


def main() -> int:
    """Build the DB and print row counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=DB_PATH, help=f"output path (default: {DB_PATH})")
    args = parser.parse_args()

    counts = build_db(args.path)
    print(f"built {args.path}:")
    for table, n in counts.items():
        print(f"  {table}: {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
