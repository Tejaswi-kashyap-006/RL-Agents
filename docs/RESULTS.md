# Results

Filled in after M8. All three policies run through the identical LangGraph harness on
the 30-task held-out validation set.

| Metric | base (untrained Qwen2.5-1.5B) | trained (GRPO) | gpt-4o-mini |
|---|---|---|---|
| Execution accuracy | — | — | — |
| Avg turns | — | — | — |
| Invalid SQL rate | — | — | — |
| Schema-first rate | — | — | — |
| Cost / latency | — | — | — |

**Execution accuracy** — % of tasks where the result set matches gold; the headline
number. **Avg turns** — efficiency; did training make it decisive? **Invalid SQL rate**
— % of `run_query` calls that error; did it learn syntax? **Schema-first rate** — % of
episodes inspecting schema before querying; did it learn procedure? **Cost / latency**
— the practical column: local 1.5B vs API calls.

## Notes

(observations, failure modes, and the honest interpretation go here after the runs)
