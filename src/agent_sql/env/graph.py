"""LangGraph agent loop — THE environment.

This graph is both the RL rollout environment and the evaluation harness.
A policy is just `Callable[[list[dict]], str]`: it gets the chat transcript
and returns raw assistant text. Untrained Qwen, GRPO-trained Qwen, and
GPT-4o-mini all plug in here unchanged — that is what makes the baseline
comparison apples-to-apples.

Text protocol (identical for every policy — no native tool-calling APIs,
because the 1.5B model must be able to emit it as plain text):

    <tool_call>{"name": "run_query", "args": {"sql": "SELECT ..."}}</tool_call>
    <answer>final answer</answer>

Termination: final answer | turn limit (MAX_TURNS) | repeated identical
tool call (degenerate loop) | non-SELECT attempt (hard fail per spec §5).
"""

import json
import re
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_sql.config import MAX_TURNS
from agent_sql.tools.sql_tools import SqlToolkit

Policy = Callable[[list[dict[str, str]]], str]

SYSTEM_PROMPT = f"""You are a SQL agent. Answer the user's question about a SQLite database.

Tools:
- list_tables() -> table names
- describe_table(name) -> columns and types
- run_query(sql) -> rows (ONE read-only SELECT, max 100 rows)

Every turn, reply with EXACTLY ONE of:
<tool_call>{{"name": "list_tables", "args": {{}}}}</tool_call>
<tool_call>{{"name": "describe_table", "args": {{"name": "products"}}}}</tool_call>
<tool_call>{{"name": "run_query", "args": {{"sql": "SELECT ..."}}}}</tool_call>
<answer>...</answer>

Rules:
- Inspect the schema (describe_table) before querying a table.
- You have at most {MAX_TURNS} turns total; the <answer> turn counts as one.
- Never repeat the exact same tool call twice.
- <answer> must contain ONLY the result: one row per line, columns separated by " | ".
  Example: <answer>Alice | 42</answer>. No explanations inside the tags."""

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_TOOL_ARG_SPECS: dict[str, dict[str, type]] = {
    "list_tables": {},
    "describe_table": {"name": str},
    "run_query": {"sql": str},
}


class AgentState(TypedDict):
    """Full episode state; the final state IS the trajectory record."""

    messages: list[dict[str, str]]
    turns_used: int
    pending_call: dict[str, Any] | None
    final_answer: str | None
    termination: str | None  # answer | turn_limit | degenerate_loop | non_select
    tool_calls: list[dict[str, Any]]  # executed calls: {name, args, observation}


def parse_action(text: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Parse one assistant message -> (tool_call, answer, error).

    Exactly one of the three is non-None. <answer> wins if both appear.
    """
    answer_match = _ANSWER_RE.search(text)
    if answer_match:
        return None, answer_match.group(1).strip(), None
    call_match = _TOOL_CALL_RE.search(text)
    if call_match:
        call_text = call_match.group(1)
    else:
        # Models routinely drop the closing tag; be lenient and take
        # everything after the opening tag (raw_decode ignores trailing junk).
        open_idx = text.find("<tool_call>")
        if open_idx == -1:
            return None, None, "no <tool_call> or <answer> tag found."
        call_text = text[open_idx + len("<tool_call>") :]
    try:
        call, _ = json.JSONDecoder().raw_decode(call_text.strip())
    except json.JSONDecodeError as e:
        return None, None, f"tool_call is not valid JSON ({e.msg})."
    if not isinstance(call, dict):
        return None, None, "tool_call must be a JSON object."
    name = call.get("name")
    if name not in _TOOL_ARG_SPECS:
        return None, None, f"unknown tool '{name}'. Tools: {', '.join(_TOOL_ARG_SPECS)}."
    args = call.get("args", {})
    spec = _TOOL_ARG_SPECS[name]
    if not isinstance(args, dict) or set(args) != set(spec) or any(
        not isinstance(args[k], t) for k, t in spec.items()
    ):
        expected = ", ".join(f'"{k}": {t.__name__}' for k, t in spec.items()) or "none"
        return None, None, f"bad args for {name}; expected: {{{expected}}}."
    return {"name": name, "args": args}, None, None


def _call_signature(call: dict[str, Any]) -> str:
    return f"{call['name']}:{json.dumps(call['args'], sort_keys=True)}"


def build_graph(policy: Policy, toolkit: SqlToolkit, max_turns: int = MAX_TURNS) -> Any:
    """Compile the agent loop for one (policy, toolkit) pair."""

    def policy_node(state: AgentState) -> dict[str, Any]:
        text = policy(state["messages"])
        messages = state["messages"] + [{"role": "assistant", "content": text}]
        turns = state["turns_used"] + 1
        update: dict[str, Any] = {
            "messages": messages,
            "turns_used": turns,
            "pending_call": None,
        }
        call, answer, error = parse_action(text)
        if answer is not None:
            update["final_answer"] = answer
            update["termination"] = "answer"
        elif turns >= max_turns:
            # A tool call on the last turn is pointless — there is no turn
            # left to answer with. Don't execute it.
            update["termination"] = "turn_limit"
        elif call is not None:
            update["pending_call"] = call
        else:
            update["messages"] = messages + [
                {
                    "role": "user",
                    "content": f"FORMAT ERROR: {error} Reply with exactly one "
                    "<tool_call>...</tool_call> or <answer>...</answer>.",
                }
            ]
        return update

    def tools_node(state: AgentState) -> dict[str, Any]:
        call = state["pending_call"]
        assert call is not None
        seen = {_call_signature(c) for c in state["tool_calls"]}
        if _call_signature(call) in seen:
            return {"pending_call": None, "termination": "degenerate_loop"}

        name, args = call["name"], call["args"]
        if name == "list_tables":
            observation = ", ".join(toolkit.list_tables())
        elif name == "describe_table":
            observation = toolkit.describe_table(args["name"])
        else:
            observation = toolkit.run_query(args["sql"])

        update: dict[str, Any] = {
            "pending_call": None,
            "tool_calls": state["tool_calls"]
            + [{"name": name, "args": args, "observation": observation}],
            "messages": state["messages"]
            + [{"role": "user", "content": f"Observation:\n{observation}"}],
        }
        # Write attempts are a hard fail: terminate immediately (spec §5).
        if observation.startswith(("ERROR[not_select]", "ERROR[multi_statement]")):
            update["termination"] = "non_select"
        return update

    def route_after_policy(state: AgentState) -> str:
        if state["termination"] is not None:
            return END
        if state["pending_call"] is not None:
            return "tools"
        return "policy"  # format error — nudge appended, try again

    def route_after_tools(state: AgentState) -> str:
        return END if state["termination"] is not None else "policy"

    graph = StateGraph(AgentState)
    graph.add_node("policy", policy_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "policy")
    graph.add_conditional_edges("policy", route_after_policy, ["tools", "policy", END])
    graph.add_conditional_edges("tools", route_after_tools, ["policy", END])
    return graph.compile()


def run_episode(
    policy: Policy,
    toolkit: SqlToolkit,
    question: str,
    max_turns: int = MAX_TURNS,
) -> AgentState:
    """Run one full episode; the returned final state is the trajectory."""
    initial: AgentState = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "turns_used": 0,
        "pending_call": None,
        "final_answer": None,
        "termination": None,
        "tool_calls": [],
    }
    graph = build_graph(policy, toolkit, max_turns)
    # Worst case ~2 nodes per turn plus format-error retries.
    return graph.invoke(initial, config={"recursion_limit": 6 * max_turns + 10})
