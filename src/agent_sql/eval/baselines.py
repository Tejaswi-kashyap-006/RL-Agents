"""Baseline policies. The untrained/trained Qwen policies arrive with M6+."""

from openai import OpenAI

from agent_sql.config import BASELINE_MODEL


class OpenAIPolicy:
    """An OpenAI chat model (default gpt-4o-mini) as the policy.

    Uses the same plain-text tool protocol as every other policy — no
    native function calling — so the comparison stays apples-to-apples.
    """

    def __init__(self, model: str = BASELINE_MODEL, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self._client = OpenAI()  # reads OPENAI_API_KEY from the environment

    def __call__(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=400,
        )
        return response.choices[0].message.content or ""
