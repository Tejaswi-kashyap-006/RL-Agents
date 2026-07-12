"""M2 tests: DB build determinism + all four run_query guards.

The read-only guard gets adversarial cases (comment-prefixed DDL,
multi-statement smuggling) because a write that slips through would
corrupt every reward computed afterwards.
"""

import sqlite3
from pathlib import Path

import pytest

from agent_sql.db.build import build_db
from agent_sql.tools.sql_tools import SqlToolkit

TABLES = ["categories", "customers", "order_items", "orders", "products"]


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("db") / "ecommerce.db"
    build_db(path)
    return path


@pytest.fixture()
def toolkit(db_path: Path) -> SqlToolkit:
    tk = SqlToolkit(db_path)
    yield tk
    tk.close()


def _dump_all(path: Path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(path)
    try:
        return {t: conn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in TABLES}
    finally:
        conn.close()


# --- build ---


def test_build_is_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    build_db(a)
    build_db(b)
    assert _dump_all(a) == _dump_all(b)


def test_build_row_counts(db_path: Path) -> None:
    dump = _dump_all(db_path)
    assert len(dump["categories"]) == 8
    assert len(dump["products"]) == 40
    assert len(dump["customers"]) == 50
    assert len(dump["orders"]) == 200
    assert len(dump["order_items"]) > 100  # must exceed the row cap for capping tests


# --- list_tables / describe_table ---


def test_list_tables(toolkit: SqlToolkit) -> None:
    assert toolkit.list_tables() == TABLES


def test_describe_table(toolkit: SqlToolkit) -> None:
    desc = toolkit.describe_table("products")
    assert desc.startswith("products:")
    assert "price REAL" in desc
    assert "category_id INTEGER" in desc


def test_describe_unknown_table_is_structured(toolkit: SqlToolkit) -> None:
    msg = toolkit.describe_table("nonexistent")
    assert msg.startswith("ERROR[unknown_table]")
    for t in TABLES:  # error must teach the valid options
        assert t in msg


# --- run_query: happy path ---


def test_basic_select(toolkit: SqlToolkit) -> None:
    out = toolkit.run_query("SELECT count(*) FROM customers")
    assert "50" in out
    assert "(1 rows)" in out


def test_cte_select_allowed(toolkit: SqlToolkit) -> None:
    out = toolkit.run_query(
        "WITH t AS (SELECT id FROM customers) SELECT count(*) FROM t"
    )
    assert "50" in out
    assert not out.startswith("ERROR")


def test_join_query(toolkit: SqlToolkit) -> None:
    out = toolkit.run_query(
        "SELECT c.name, count(o.id) FROM customers c "
        "JOIN orders o ON o.customer_id = c.id GROUP BY c.id LIMIT 3"
    )
    assert not out.startswith("ERROR")
    assert "(3 rows)" in out


# --- guard: read-only ---


WRITE_ATTEMPTS = [
    "INSERT INTO customers VALUES (999, 'x', 'x@x.com', 'X', '2024-01-01')",
    "UPDATE customers SET name = 'hacked'",
    "DELETE FROM orders",
    "DROP TABLE customers",
    "CREATE TABLE evil (id INTEGER)",
    "ALTER TABLE customers ADD COLUMN evil TEXT",
    "PRAGMA writable_schema = 1",
    "ATTACH DATABASE ':memory:' AS evil",
    "  \n\t insert into customers values (999, 'x', 'y@x.com', 'X', '2024-01-01')",
    "-- just a comment\nDROP TABLE customers",
    "/* SELECT */ DELETE FROM orders",
    "",
]


@pytest.mark.parametrize("sql", WRITE_ATTEMPTS)
def test_non_select_rejected(toolkit: SqlToolkit, sql: str) -> None:
    out = toolkit.run_query(sql)
    assert out.startswith("ERROR[not_select]")


def test_multi_statement_rejected(toolkit: SqlToolkit) -> None:
    out = toolkit.run_query("SELECT 1; DROP TABLE customers")
    assert out.startswith("ERROR[multi_statement]")


def test_data_unchanged_after_attacks(toolkit: SqlToolkit, db_path: Path) -> None:
    """Run after the attack tests in this file: the data must be intact."""
    for sql in WRITE_ATTEMPTS:
        toolkit.run_query(sql)
    toolkit.run_query("SELECT 1; DROP TABLE customers")
    dump = _dump_all(db_path)
    assert len(dump["customers"]) == 50
    assert len(dump["orders"]) == 200
    assert not any(r[1] == "hacked" for r in dump["customers"])


def test_readonly_at_engine_level(db_path: Path) -> None:
    """Defense in depth: even the raw connection must refuse writes."""
    tk = SqlToolkit(db_path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        tk._conn.execute("DELETE FROM orders")
    tk.close()


# --- guard: row cap ---


def test_row_cap(toolkit: SqlToolkit) -> None:
    out = toolkit.run_query("SELECT * FROM order_items")
    lines = out.splitlines()
    assert "truncated at the row cap" in lines[-1]
    assert len(lines) == 1 + toolkit.row_cap + 1  # header + capped rows + note


def test_under_cap_not_truncated(toolkit: SqlToolkit) -> None:
    out = toolkit.run_query("SELECT * FROM categories")
    assert "(8 rows)" in out
    assert "truncated" not in out


# --- guard: structured errors ---


def test_bad_column_error_is_actionable(toolkit: SqlToolkit) -> None:
    out = toolkit.run_query("SELECT nonexistent_col FROM customers")
    assert out.startswith("ERROR[sql]")
    assert "no such column" in out
    assert "describe_table" in out  # tells the agent how to self-correct


def test_bad_table_error_is_actionable(toolkit: SqlToolkit) -> None:
    out = toolkit.run_query("SELECT * FROM no_such_table")
    assert out.startswith("ERROR[sql]")
    assert "no such table" in out


# --- guard: timeout ---


def test_timeout(db_path: Path) -> None:
    tk = SqlToolkit(db_path, timeout_s=0.2)
    out = tk.run_query(
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT max(x) FROM c"
    )
    tk.close()
    assert out.startswith("ERROR[timeout]")
