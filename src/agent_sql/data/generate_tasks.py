"""Synthesize NL questions with gold SQL via OpenAI, verified against the DB.

Every candidate task is validated by actually executing its gold SQL on a
read-only connection. Quality gates (beyond "it runs"):

  - single SELECT/WITH statement only
  - result is 1-10 rows and 1-3 columns — the trained 1.5B model must fit
    its entire answer inside a tight completion budget, so tasks whose
    answers are large tables are useless for training
  - no NULLs in the result (ambiguous to format in an answer)
  - deduplicated by normalized question and normalized SQL
"""

import json
import sqlite3
from pathlib import Path
from random import Random

from openai import OpenAI

from agent_sql.config import DB_SEED, N_TASKS, N_TRAIN, TASK_GEN_MODEL
from agent_sql.data.tasks import Task

MAX_ROWS = 10
MAX_COLS = 3
_BATCH_SIZE = 20
_MAX_BATCHES = 30

_SYSTEM = """You generate evaluation tasks for a text-to-SQL agent working on SQLite.
Each task is a natural-language question plus the ONE correct SQL query (the gold SQL).
Output strict JSON: {"tasks": [{"question": "...", "sql": "..."}, ...]}"""

_REQUIREMENTS = f"""Requirements for every task:
- SQLite dialect. A single SELECT statement (WITH is allowed). No comments.
- The question must be answerable from this database alone, with exactly one
  correct result. Phrase it unambiguously (name the exact status, city,
  category, year etc. you mean).
- The result must have 1-{MAX_ROWS} rows and 1-{MAX_COLS} columns, with no NULL values.
- For "most / top / highest" questions, use ORDER BY ... LIMIT and make the
  question imply the row count (e.g. "top 3 ...").
- Vary difficulty across the batch: single-table filters and counts,
  GROUP BY aggregates, 2-table joins, 3+ table joins, date-range questions.
- Do not ask for whole-table dumps or open-ended lists.
"""


def _schema_and_samples(conn: sqlite3.Connection) -> str:
    """Schema DDL plus a few sample rows per table, for the generator prompt."""
    parts: list[str] = []
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    for t in tables:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (t,)
        ).fetchone()[0]
        parts.append(ddl)
        rows = conn.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()
        parts.append(f"-- sample rows: {rows}")
    return "\n".join(parts)


def _validate_sql(conn: sqlite3.Connection, sql: str) -> str | None:
    """Return None if the gold SQL passes all gates, else the reason."""
    sql = sql.strip().rstrip(";").strip()
    first = sql.split(None, 1)[0].upper() if sql.split() else ""
    if first not in ("SELECT", "WITH"):
        return f"not a SELECT (starts with '{first}')"
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        return f"does not execute: {e}"
    if not rows:
        return "empty result"
    if len(rows) > MAX_ROWS:
        return f"more than {MAX_ROWS} rows"
    n_cols = len(cursor.description)
    if n_cols > MAX_COLS:
        return f"{n_cols} columns > {MAX_COLS}"
    if any(v is None for row in rows for v in row):
        return "NULL in result"
    return None


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _request_batch(
    client: OpenAI, model: str, context: str, avoid: list[str]
) -> list[dict[str, str]]:
    avoid_text = ""
    if avoid:
        joined = "\n".join(f"- {q}" for q in avoid[-60:])
        avoid_text = f"\nAlready used (do NOT repeat or lightly rephrase):\n{joined}\n"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": f"Database:\n{context}\n\n{_REQUIREMENTS}{avoid_text}\n"
                f"Generate {_BATCH_SIZE} tasks as JSON.",
            },
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    try:
        payload = json.loads(response.choices[0].message.content or "{}")
        batch = payload.get("tasks", [])
    except json.JSONDecodeError:
        return []
    return [
        t for t in batch
        if isinstance(t, dict) and isinstance(t.get("question"), str)
        and isinstance(t.get("sql"), str)
    ]


def generate_tasks(
    db_path: str | Path,
    n_tasks: int = N_TASKS,
    n_train: int = N_TRAIN,
    model: str = TASK_GEN_MODEL,
) -> list[Task]:
    """Generate `n_tasks` verified tasks; raises if the quota can't be met."""
    conn = sqlite3.connect(
        f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True
    )
    client = OpenAI()
    context = _schema_and_samples(conn)

    accepted: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    seen_sql: set[str] = set()
    rejected = 0

    for batch_i in range(_MAX_BATCHES):
        if len(accepted) >= n_tasks:
            break
        batch = _request_batch(
            client, model, context, [t["question"] for t in accepted]
        )
        for cand in batch:
            if len(accepted) >= n_tasks:
                break
            question = cand["question"].strip()
            sql = cand["sql"].strip().rstrip(";").strip()
            if _normalize(question) in seen_questions or _normalize(sql) in seen_sql:
                rejected += 1
                continue
            reason = _validate_sql(conn, sql)
            if reason is not None:
                rejected += 1
                continue
            seen_questions.add(_normalize(question))
            seen_sql.add(_normalize(sql))
            accepted.append({"question": question, "sql": sql})
        print(
            f"batch {batch_i + 1}: accepted {len(accepted)}/{n_tasks} "
            f"(rejected so far: {rejected})"
        )

    conn.close()
    if len(accepted) < n_tasks:
        raise RuntimeError(
            f"only {len(accepted)}/{n_tasks} valid tasks after {_MAX_BATCHES} batches "
            f"({rejected} rejected). Loosen the gates or raise _MAX_BATCHES."
        )

    Random(DB_SEED).shuffle(accepted)
    return [
        Task(
            id=f"t{i + 1:03d}",
            question=t["question"],
            gold_sql=t["sql"],
            split="train" if i < n_train else "val",
        )
        for i, t in enumerate(accepted)
    ]


def verify_tasks(tasks: list[Task], db_path: str | Path) -> list[tuple[str, str]]:
    """Re-run every task's gold SQL through the gates; return (id, reason) failures."""
    conn = sqlite3.connect(
        f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True
    )
    failures = []
    for t in tasks:
        reason = _validate_sql(conn, t.gold_sql)
        if reason is not None:
            failures.append((t.id, reason))
    conn.close()
    return failures
