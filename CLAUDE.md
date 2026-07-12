# CLAUDE.md

CLAUDE.md — GRPO-Trained SQL Agent


Build spec for Claude Code. Read this file fully before writing any code.
Work milestone by milestone. Commit after each. Do not skip the smoke tests.




0. Project

Train a small open-weights LLM to be a competent multi-turn, tool-using SQL agent using GRPO with verifiable rewards. Compare it against a prompted GPT-4o-mini baseline running through the same LangGraph harness.

The thesis under test: can a 1.5B model trained to be an agent beat a frontier model prompted to be an agent, on a narrow, verifiable task?

Why this task: SQL is genuinely multi-turn (the agent must explore the schema before it can query), and genuinely verifiable (execute the gold SQL, compare result sets). No LLM judge required. That combination is what makes RL viable here.


1. READ FIRST — Hardware Constraints & Safety

Target machine: laptop RTX 4060, 8 GB VRAM, 16 GB system RAM, Windows + CUDA. Every decision below is downstream of that.

Non-negotiable safety rules — implement these, they are not optional polish


Preflight gate. scripts/preflight.py must verify: CUDA visible, free VRAM >= 7.0 GB, GPU temp < 70 C at idle, AC power connected. Abort with a clear message on any failure. No training script may run without it passing.
Thermal watchdog. Poll nvidia-smi every 10 s during training. If GPU temp exceeds 85 C continuously for 60 s, checkpoint and halt gracefully. Log it loudly.
Checkpoint every 25 steps. The developer must be able to Ctrl-C at any moment and lose at most a few minutes.
Default to the smoke tier. Any script run without --tier full runs a ~10-step config finishing in under 10 minutes. Never let a misconfigured run burn three hours before failing.
Document a power cap in docs/HARDWARE.md (do not enforce in code): nvidia-smi -pl <watts> caps board power, cutting heat for a modest speed cost. Explain how to read the default and safe range.


Why these matter — put this in the README, plainly

The GPU cannot be physically damaged by sustained load. Driver and firmware enforce thermal/power limits; the card throttles, and worst case the process is killed. Nothing melts.

The genuine risks are: (a) an OOM crash (harmless, but it will happen), (b) sustained thermal stress on a laptop chassis over a multi-hour run — fans at full tilt, hot chassis, and (c) losing hours of work to an unrecoverable crash. The guards above address all three. Never train on battery.

Memory budget — do not exceed

ComponentBudgetPolicy model (Qwen2.5-1.5B-Instruct, QLoRA 4-bit)~2.0 GBvLLM rollout engine (gpu_memory_utilization=0.55)~3.5 GBLoRA grads + optimizer states + activations~1.5 GBHeadroom for Windows display + spikes~1.0 GBTotal~8.0 GB — at the limit

OOM ladder (apply in this order, document in README): reduce num_generations (6 -> 4) -> reduce max_seq_length -> reduce gpu_memory_utilization -> reduce lora_rank.


2. Environment Setup — WSL2 is REQUIRED

vLLM does not run natively on Windows. GRPO depends on vLLM for rollout generation. Therefore this project runs inside WSL2 (Ubuntu 22.04) with NVIDIA GPU passthrough. This is the single biggest setup trap — treat docs/SETUP.md as a first-class deliverable.

docs/SETUP.md must cover, in order:


Enable WSL2; install Ubuntu 22.04.
Install the Windows-side NVIDIA driver ONLY. Do not install a driver inside WSL — passthrough handles it. Verify with nvidia-smi run inside WSL.
Create ~/.wslconfig on the Windows side capping WSL memory to 12 GB (leaves 4 GB for Windows; prevents host thrashing on a 16 GB machine).
Install CUDA toolkit inside WSL, then uv, then pinned deps.
Run python scripts/preflight.py; confirm all checks pass.


Escape hatches — document both. WSL + Unsloth installs are notoriously fragile (bitsandbytes/numpy/torch version conflicts are common). If the install fights back: (a) use Unsloth's official Docker image, which supports Windows/WSL/Linux, or (b) run training in Colab and only run inference/eval locally.


3. Repository Structure

Build exactly this. Modular and testable — no monolithic scripts.

grpo-sql-agent/
├── README.md
├── CLAUDE.md                    # this file
├── pyproject.toml
├── .env.example                 # OPENAI_API_KEY=
├── .gitignore
├── docs/
│   ├── SETUP.md                 # WSL2 + CUDA + deps, with escape hatches
│   ├── HARDWARE.md              # 8GB budget, OOM ladder, thermal guidance
│   └── RESULTS.md               # the money table; filled in after M8
├── src/agent_sql/
│   ├── __init__.py
│   ├── config.py                # ALL hyperparams; tiers: smoke | full
│   ├── db/
│   │   ├── build.py             # generate + seed the SQLite DB
│   │   └── schema.sql
│   ├── tools/
│   │   ├── __init__.py
│   │   └── sql_tools.py         # list_tables, describe_table, run_query
│   ├── env/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph agent loop == THE environment
│   │   └── rollout.py           # graph -> trajectory (see §5, trickiest file)
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── verifiable.py        # execution-match: the core RLVR signal
│   │   ├── shaping.py           # format, efficiency, schema-discipline
│   │   └── hacking_guards.py    # anti-reward-hacking
│   ├── data/
│   │   ├── tasks.py             # Task dataclass; train/val split
│   │   └── generate_tasks.py    # uses OpenAI to synthesize NL questions
│   ├── train/
│   │   ├── grpo.py              # Unsloth + TRL GRPOTrainer
│   │   └── thermal_guard.py     # the watchdog
│   ├── eval/
│   │   ├── harness.py           # runs ANY policy through the SAME graph
│   │   └── baselines.py         # gpt-4o-mini + untrained-Qwen
│   └── utils/
│       ├── gpu.py               # VRAM/temp probes via nvidia-smi
│       └── logging.py
├── scripts/
│   ├── preflight.py             # SAFETY GATE — run before anything
│   ├── build_db.py
│   ├── gen_tasks.py
│   ├── train.py                 # --tier smoke|full
│   ├── evaluate.py              # --policy base|trained|gpt-4o-mini
│   └── watch_gpu.sh             # live temp/VRAM monitor
└── tests/
    ├── test_tools.py
    ├── test_rewards.py          # THE MOST IMPORTANT TESTS — see §4
    └── test_env.py


4. The Environment (LangGraph)

Core design principle: the LangGraph agent loop is both the RL rollout environment and the evaluation harness. Any policy — untrained Qwen, GRPO-trained Qwen, or GPT-4o-mini — plugs into the same graph. This is what makes the baseline comparison apples-to-apples, and it is the architectural heart of the project.

The database

A small synthetic e-commerce SQLite DB, deterministic seed: customers, orders, order_items, products, categories. Large enough that real questions need joins; small enough that every query returns instantly.

The tools

pythonlist_tables()              -> list[str]
describe_table(name: str)  -> str    # column names + types
run_query(sql: str)        -> str    # rows, or a STRUCTURED error

run_query must be:


Read-only — reject anything that is not a SELECT.
Row-capped — inject LIMIT 100.
Timeout-guarded — 2 s.
Structured on error — return actionable messages the agent can self-correct from. This is what makes multi-turn learning possible at all; a bare "error" teaches nothing.


The loop

task ──> [POLICY] ──> tool call ──> [TOOLS] ──> observation ──┐
            ▲                                                  │
            └──────────────────────────────────────────────────┘
                          (max 6 turns)
                              │
                              ▼
                    final answer ──> [REWARD]

Terminate on: final answer, turn limit (6), or repeated identical tool calls (a common degenerate loop).


5. The Reward Function — the most important code in the repo

This is RLVR: the reward is programmatically verifiable. No LLM judge.

Primary signal — execution match

pythondef execution_match(predicted_answer, gold_sql, db) -> float:
    """Execute gold SQL; compare the agent's final answer to the result set."""
    gold_rows = db.execute(gold_sql).fetchall()
    # Compare as SETS of tuples — order-insensitive unless the question implies ordering.
    # 1.0 on exact match, 0.0 otherwise. NO partial credit.

Gold SQL is stored with each task. The agent never sees it.

Shaping terms — small, capped, and gated

total = 1.0 * execution_match + shaping, where shaping is capped at 0.3 and paid only when execution_match == 1.0.


format_reward (0.1) — final answer inside <answer>...</answer>.
efficiency_reward (0.1) — 0.1 * (max_turns - turns_used) / max_turns.
schema_discipline (0.1) — called describe_table before querying that table.


Anti-reward-hacking guards (hacking_guards.py)

These separate a real project from a toy. Implement and test each:


Gate auxiliary rewards on success. No shaping unless the answer is correct — otherwise the agent learns to be fast and wrong.
Defeat SELECT * dumps. An agent may try to "answer" by dumping a whole table. Strict set-comparison must score this 0.
Detect degenerate loops. Repeated identical tool calls -> terminate, reward 0.
Reject non-SELECT. Any DDL/DML attempt -> immediate 0, terminate.


test_rewards.py must include adversarial trajectories that actively try to hack the reward, asserting each scores 0. A broken reward function is the #1 way this project silently fails — you will train for hours and learn garbage. Test this harder than anything else.


6. Training (GRPO)

Use Unsloth + TRL GRPOTrainer. Unsloth is required, not optional — stock TRL will not fit GRPO into 8 GB.

Config (config.py)

python# --- Model ---
MODEL = "unsloth/Qwen2.5-1.5B-Instruct"   # 1.5B is the practical floor:
                                           # smaller models don't reliably emit reasoning
LOAD_IN_4BIT = True                        # QLoRA — required at 8GB
LORA_RANK = 16                             # 32 if VRAM allows; 16 is the safe start
MAX_SEQ_LENGTH = 1536                      # prompt + completion. Keep tight.
GPU_MEMORY_UTILIZATION = 0.55              # vLLM's share. Conservative on a laptop.

# --- GRPO ---
# The GROUP is the baseline. Advantage = (r - mean) / std across the group.
# With num_generations = 1, std = 0 and the advantage is UNDEFINED.
# NEVER set this below 2.
NUM_GENERATIONS = 6                        # drop to 4 on OOM

TIERS = {
    "smoke": dict(max_steps=10,  save_steps=5,  num_generations=4),
    "full":  dict(max_steps=300, save_steps=25, num_generations=6),
}

# --- Optimizer (known-good GRPO values) ---
LEARNING_RATE = 5e-6
OPTIM = "paged_adamw_8bit"
MAX_GRAD_NORM = 0.1
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

Expectations — write these into the README honestly


Reward stays flat for 100–200 steps, then climbs. This is normal GRPO behavior. Do NOT conclude it's broken at step 30.
A meaningful run is 300+ steps and can take several hours.
Run --tier smoke first to verify the pipeline. The smoke tier is not expected to learn anything — it exists to prove the reward plumbing works and nothing OOMs.


Multi-turn rollout — the trickiest part of the codebase

TRL's GRPOTrainer is built around single-completion generation. For multi-turn tool use, the rollout must be flattened: run the full LangGraph episode, then present the entire trajectory (all assistant turns concatenated, tool observations as context) as the "completion" to be scored, with the episode's total scalar reward.

Document this explicitly in rollout.py with a worked example. It is where a reader — and you — will get confused. Note the limitation honestly in the README: this gives episode-level credit assignment, not per-turn. Every turn in a successful episode is reinforced equally, including the wasteful ones. Per-turn credit assignment is the open research problem; do not pretend to solve it.


7. Evaluation & the Baseline Comparison

scripts/evaluate.py --policy {base|trained|gpt-4o-mini} runs the held-out task set through the same LangGraph harness.

MetricMeaningExecution accuracy% tasks where the result set matches gold. The headline number.Avg turnsEfficiency. Did training make it decisive?Invalid SQL rate% of run_query calls that error. Did it learn syntax?Schema-first rate% episodes inspecting schema before querying. Did it learn procedure?Cost / latencyThe practical column: local 1.5B vs API calls.

The gpt-4o-mini baseline is the entire point — it answers the project's question empirically. Write results into docs/RESULTS.md as a table.

Be honest if the trained model loses. That is a finding, not a failure, and it is the most interesting possible result: it would mean scaffolding a frontier model beats training a small one on this task, which is exactly what most teams should hear.


8. Git & Workflow


git init at the start. Commit after each milestone.
Conventional commits: feat:, fix:, docs:, test:.
.gitignore: .env, outputs/, *.db, checkpoints/, __pycache__/, .venv/.
Never commit the API key. .env.example only.


Build order — do NOT skip ahead

Each milestone must run and pass its tests before the next begins.


M1 — Scaffolding. Repo structure, pyproject.toml, config.py, preflight.py. python scripts/preflight.py runs and reports GPU status honestly.
M2 — DB + tools. build_db.py works; test_tools.py passes; read-only guard proven.
M3 — Environment. LangGraph loop runs end-to-end with gpt-4o-mini as the policy. You now have a working agent before any RL. Commit — this is the baseline.
M4 — Tasks. 150 tasks (120 train / 30 val), each with gold SQL, each verified to execute.
M5 — Rewards. test_rewards.py passes, including every adversarial hacking test. Do not proceed until the reward is trustworthy.
M6 — Train (smoke). python scripts/train.py --tier smoke completes in <10 min, no OOM, reward logged and non-degenerate (varies across the group).
M7 — Train (full). Long run. Thermal guard + checkpointing active.
M8 — Evaluate. All three policies through the harness. RESULTS.md written.



9. Rules for You (Claude Code)


Ask before installing anything heavy. vLLM/Unsloth installs are large and fragile — confirm WSL is live and nvidia-smi works inside it first.
Never launch a full training run unprompted. Smoke tier only, unless explicitly asked.
Fail loudly, not silently. A crash with a clear message beats four hours of training on a broken reward.
If VRAM is tight, say so. Do not silently downgrade settings to make something fit — surface the tradeoff and let the developer choose.
Test the reward function harder than anything else. Everything else is plumbing; the reward is the science.
Type hints everywhere. Docstrings on public functions. ruff clean.


## Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
