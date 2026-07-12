"""The agent's three tools: list_tables, describe_table, run_query.

run_query guards (all four are load-bearing for RL):
  - Read-only: first keyword must be SELECT (or WITH for CTEs), AND the
    connection is opened with SQLite's mode=ro — defense in depth, so a
    guard bypass still cannot write.
  - Row-capped: at most QUERY_ROW_CAP rows are returned. Enforced at fetch
    time (fetch cap+1, truncate) rather than by rewriting the SQL with an
    injected LIMIT — same effect, but robust to CTEs and ORDER BY.
  - Timeout-guarded: a progress handler aborts queries after QUERY_TIMEOUT_S.
  - Structured errors: every failure returns "ERROR[<code>]: <message>" with
    an actionable hint the agent can self-correct from. A bare "error"
    teaches nothing.
"""

import sqlite3
import time
from pathlib import Path

from agent_sql.config import QUERY_ROW_CAP, QUERY_TIMEOUT_S

_ALLOWED_FIRST_KEYWORDS = ("SELECT", "WITH")
_PROGRESS_HANDLER_INSTRUCTIONS = 5000  # how often (in VM ops) the timeout check runs


def _strip_leading_comments(sql: str) -> str:
    """Remove leading whitespace and SQL comments so the guard sees the real
    first keyword (defeats '/* hi */ DROP ...')."""
    s = sql.lstrip()
    while True:
        if s.startswith("--"):
            newline = s.find("\n")
            s = "" if newline == -1 else s[newline + 1 :].lstrip()
        elif s.startswith("/*"):
            end = s.find("*/")
            s = "" if end == -1 else s[end + 2 :].lstrip()
        else:
            return s


class SqlToolkit:
    """Read-only tool surface over one SQLite database.

    One instance per episode is fine; the connection is opened in
    read-only mode and never holds a transaction.
    """

    def __init__(
        self,
        db_path: str | Path,
        row_cap: int = QUERY_ROW_CAP,
        timeout_s: float = QUERY_TIMEOUT_S,
    ) -> None:
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {db_path}. Run: python scripts/build_db.py"
            )
        self.row_cap = row_cap
        self.timeout_s = timeout_s
        # mode=ro makes writes impossible at the engine level, regardless
        # of what slips past the keyword guard.
        self._conn = sqlite3.connect(
            f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True
        )

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def list_tables(self) -> list[str]:
        """Names of all user tables, sorted."""
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def describe_table(self, name: str) -> str:
        """Column names and types for `name`, or a structured error."""
        tables = self.list_tables()
        if name not in tables:
            return (
                f"ERROR[unknown_table]: no table named '{name}'. "
                f"Valid tables: {', '.join(tables)}."
            )
        cols = self._conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        col_desc = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        return f"{name}: {col_desc}"

    def run_query(self, sql: str) -> str:
        """Execute a single SELECT and return rows as text, or ERROR[...].

        Error codes: not_select, multi_statement, timeout, sql.
        """
        stripped = _strip_leading_comments(sql)
        first_word = stripped.split(None, 1)[0].upper() if stripped.split() else ""
        first_word = first_word.rstrip("(;")
        if first_word not in _ALLOWED_FIRST_KEYWORDS:
            return (
                f"ERROR[not_select]: only SELECT queries are allowed "
                f"(got '{first_word or 'empty statement'}'). "
                "This database is read-only."
            )

        deadline = time.monotonic() + self.timeout_s
        self._conn.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0,
            _PROGRESS_HANDLER_INSTRUCTIONS,
        )
        try:
            cursor = self._conn.execute(stripped)
            rows = cursor.fetchmany(self.row_cap + 1)
            columns = [d[0] for d in cursor.description] if cursor.description else []
        except sqlite3.Warning:
            return (
                "ERROR[multi_statement]: only ONE statement per call. "
                "Remove the ';' and send a single SELECT."
            )
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                return (
                    f"ERROR[timeout]: query exceeded the {self.timeout_s}s limit. "
                    "Simplify it (avoid unbounded recursion / huge joins)."
                )
            return (
                f"ERROR[sql]: {e}. "
                "Check table names with list_tables() and columns with describe_table()."
            )
        except sqlite3.Error as e:
            return (
                f"ERROR[sql]: {e}. "
                "Check table names with list_tables() and columns with describe_table()."
            )
        finally:
            self._conn.set_progress_handler(None, 0)

        truncated = len(rows) > self.row_cap
        if truncated:
            rows = rows[: self.row_cap]
        lines = [" | ".join(columns)]
        lines += [" | ".join(str(v) for v in row) for row in rows]
        if truncated:
            lines.append(f"({self.row_cap} rows shown; result truncated at the row cap)")
        else:
            lines.append(f"({len(rows)} rows)")
        return "\n".join(lines)
