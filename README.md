# GRPO-Trained SQL Agent

Can a 1.5B model **trained** to be an agent beat a frontier model **prompted** to be an
agent, on a narrow, verifiable task?

This repo trains Qwen2.5-1.5B-Instruct with GRPO (verifiable rewards, no LLM judge) to be
a multi-turn, tool-using SQL agent, and compares it against a prompted GPT-4o-mini baseline
running through the **same** LangGraph harness. The task: answer natural-language questions
over a synthetic e-commerce SQLite database, using `list_tables` / `describe_table` /
`run_query` tools, in at most 6 turns.

Why SQL: it is genuinely multi-turn (the agent must explore the schema before it can
query) and genuinely verifiable (execute the gold SQL, compare result sets). That
combination is what makes RL viable here.

## Hardware safety — read before training

Target machine: laptop RTX 4060, 8 GB VRAM, 16 GB RAM.

**The GPU cannot be physically damaged by sustained load.** Driver and firmware enforce
thermal and power limits; the card throttles, and worst case the process is killed.
Nothing melts.

The genuine risks are:

- **(a) an OOM crash** — harmless, but it will happen at 8 GB;
- **(b) sustained thermal stress on a laptop chassis** over a multi-hour run — fans at
  full tilt, hot chassis;
- **(c) losing hours of work** to an unrecoverable crash.

The guards address all three:

1. **Preflight gate** — `python scripts/preflight.py` must pass before any training:
   CUDA visible, ≥ 7.0 GB VRAM free, GPU < 70 °C at idle, on AC power.
2. **Thermal watchdog** — polls every 10 s during training; > 85 °C sustained for 60 s
   → checkpoint and halt gracefully.
3. **Checkpoint every 25 steps** — Ctrl-C at any moment loses at most a few minutes.
4. **Smoke tier by default** — every script without `--tier full` runs a ~10-step config
   that finishes in under 10 minutes.

**Never train on battery.** See `docs/HARDWARE.md` for the memory budget and the optional
power cap.

### OOM ladder

If training OOMs, apply in this order (one step at a time):

1. `NUM_GENERATIONS` 6 → 4
2. reduce `MAX_SEQ_LENGTH`
3. reduce `GPU_MEMORY_UTILIZATION`
4. reduce `LORA_RANK`

## Training expectations — honest version

- **Reward stays flat for 100–200 steps, then climbs.** This is normal GRPO behavior.
  Do NOT conclude it's broken at step 30.
- A meaningful run is 300+ steps and takes several hours.
- Run `--tier smoke` first. The smoke tier is not expected to learn anything — it exists
  to prove the reward plumbing works and nothing OOMs.

### Known limitation: episode-level credit assignment

TRL's GRPOTrainer scores single completions, so multi-turn episodes are flattened: the
whole trajectory is presented as one "completion" with the episode's total reward. Every
turn in a successful episode is reinforced equally, including the wasteful ones. Per-turn
credit assignment is an open research problem; this project does not pretend to solve it.
See `src/agent_sql/env/rollout.py` for the worked example.

## Setup

Training requires WSL2 — vLLM does not run on native Windows. Follow `docs/SETUP.md`
end to end, including the escape hatches if the Unsloth install fights back.

Quick start (after setup):

```bash
python scripts/preflight.py          # safety gate — must pass
python scripts/build_db.py           # build the SQLite DB
python scripts/gen_tasks.py          # synthesize tasks (needs OPENAI_API_KEY)
python scripts/train.py --tier smoke # prove the pipeline
python scripts/evaluate.py --policy gpt-4o-mini
```

## Results

See `docs/RESULTS.md` (filled in after M8). If the trained model loses to the prompted
baseline, that is a finding, not a failure — it would mean scaffolding a frontier model
beats training a small one on this task.
