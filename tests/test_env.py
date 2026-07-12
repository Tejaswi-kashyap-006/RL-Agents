"""M3 tests: every termination path of the LangGraph loop, via scripted
policies — no API calls, no cost."""

import json

import pytest

from agent_sql.db.build import build_db
from agent_sql.env.graph import parse_action, run_episode
from agent_sql.tools.sql_tools import SqlToolkit


class ScriptedPolicy:
    """Plays back canned outputs; fails the test if called too often."""

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    def __call__(self, messages: list[dict[str, str]]) -> str:
        assert self._outputs, "policy called more times than scripted"
        self.calls += 1
        return self._outputs.pop(0)


def tool_call(tool: str, **args: str) -> str:
    return f"<tool_call>{json.dumps({'name': tool, 'args': args})}</tool_call>"


@pytest.fixture(scope="session")
def toolkit(tmp_path_factory: pytest.TempPathFactory) -> SqlToolkit:
    path = tmp_path_factory.mktemp("envdb") / "ecommerce.db"
    build_db(path)
    tk = SqlToolkit(path)
    yield tk
    tk.close()


# --- parse_action ---


def test_parse_answer() -> None:
    call, answer, err = parse_action("thinking...\n<answer>42</answer>")
    assert (call, answer, err) == (None, "42", None)


def test_parse_tool_call() -> None:
    call, answer, err = parse_action(tool_call("describe_table", name="orders"))
    assert call == {"name": "describe_table", "args": {"name": "orders"}}
    assert answer is None and err is None


def test_parse_tool_call_missing_closing_tag() -> None:
    """gpt-4o-mini drops the closing tag routinely — must still parse."""
    text = '<tool_call>{"name": "describe_table", "args": {"name": "products"}}'
    call, answer, err = parse_action(text)
    assert call == {"name": "describe_table", "args": {"name": "products"}}
    assert err is None


def test_parse_answer_wins_over_tool_call() -> None:
    text = tool_call("list_tables") + "<answer>7</answer>"
    call, answer, _ = parse_action(text)
    assert call is None and answer == "7"


@pytest.mark.parametrize(
    "bad",
    [
        "no tags at all",
        "<tool_call>not json</tool_call>",
        '<tool_call>{"name": "drop_db", "args": {}}</tool_call>',
        '<tool_call>{"name": "run_query", "args": {}}</tool_call>',  # missing sql
        '<tool_call>{"name": "run_query", "args": {"sql": 5}}</tool_call>',  # wrong type
        '<tool_call>{"name": "list_tables", "args": {"x": 1}}</tool_call>',  # extra arg
    ],
)
def test_parse_rejects_malformed(bad: str) -> None:
    call, answer, err = parse_action(bad)
    assert call is None and answer is None
    assert err  # actionable message present


# --- episode: happy path ---


def test_full_episode(toolkit: SqlToolkit) -> None:
    policy = ScriptedPolicy(
        [
            tool_call("list_tables"),
            tool_call("describe_table", name="customers"),
            tool_call("run_query", sql="SELECT count(*) FROM customers"),
            "<answer>50</answer>",
        ]
    )
    state = run_episode(policy, toolkit, "How many customers are there?")
    assert state["termination"] == "answer"
    assert state["final_answer"] == "50"
    assert state["turns_used"] == 4
    assert [c["name"] for c in state["tool_calls"]] == [
        "list_tables",
        "describe_table",
        "run_query",
    ]
    assert "50" in state["tool_calls"][2]["observation"]
    # observations reached the transcript for the next policy turn
    assert any(
        m["role"] == "user" and m["content"].startswith("Observation:")
        for m in state["messages"]
    )


# --- episode: termination paths ---


def test_turn_limit(toolkit: SqlToolkit) -> None:
    # 6 distinct valid tool calls, never answers.
    policy = ScriptedPolicy(
        [tool_call("run_query", sql=f"SELECT {i}") for i in range(1, 7)]
    )
    state = run_episode(policy, toolkit, "q")
    assert state["termination"] == "turn_limit"
    assert state["final_answer"] is None
    assert state["turns_used"] == 6
    assert len(state["tool_calls"]) == 5  # the 6th call is pointless — not executed


def test_degenerate_loop(toolkit: SqlToolkit) -> None:
    policy = ScriptedPolicy([tool_call("list_tables"), tool_call("list_tables")])
    state = run_episode(policy, toolkit, "q")
    assert state["termination"] == "degenerate_loop"
    assert len(state["tool_calls"]) == 1  # the repeat is not executed


def test_non_select_terminates(toolkit: SqlToolkit) -> None:
    policy = ScriptedPolicy([tool_call("run_query", sql="DELETE FROM orders")])
    state = run_episode(policy, toolkit, "q")
    assert state["termination"] == "non_select"
    assert state["tool_calls"][0]["observation"].startswith("ERROR[not_select]")


def test_format_error_nudges_then_recovers(toolkit: SqlToolkit) -> None:
    policy = ScriptedPolicy(["I think the answer is 50.", "<answer>50</answer>"])
    state = run_episode(policy, toolkit, "q")
    assert state["termination"] == "answer"
    assert state["turns_used"] == 2
    assert any(
        m["role"] == "user" and m["content"].startswith("FORMAT ERROR")
        for m in state["messages"]
    )


def test_sql_error_does_not_terminate(toolkit: SqlToolkit) -> None:
    """Ordinary SQL mistakes must feed back as observations, not end the
    episode — self-correction is the whole point of multi-turn."""
    policy = ScriptedPolicy(
        [
            tool_call("run_query", sql="SELECT nope FROM customers"),
            tool_call("run_query", sql="SELECT count(*) FROM customers"),
            "<answer>50</answer>",
        ]
    )
    state = run_episode(policy, toolkit, "q")
    assert state["termination"] == "answer"
    assert state["tool_calls"][0]["observation"].startswith("ERROR[sql]")
    assert state["final_answer"] == "50"
