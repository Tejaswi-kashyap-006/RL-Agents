"""All hyperparameters and constants for the project, in one place.

Tiers: "smoke" (default — ~10 steps, <10 min, proves the plumbing) and
"full" (300 steps, several hours, the real run). Any script that takes
--tier must default to smoke; never let a misconfigured run burn hours.
"""

from typing import Any

# --- Model ---
MODEL = "unsloth/Qwen2.5-1.5B-Instruct"  # 1.5B is the practical floor:
#                                          smaller models don't reliably emit reasoning
LOAD_IN_4BIT = True  # QLoRA — required at 8GB
LORA_RANK = 16  # 32 if VRAM allows; 16 is the safe start
MAX_SEQ_LENGTH = 1536  # prompt + completion. Keep tight.
GPU_MEMORY_UTILIZATION = 0.55  # vLLM's share. Conservative on a laptop.

# --- GRPO ---
# The GROUP is the baseline. Advantage = (r - mean) / std across the group.
# With num_generations = 1, std = 0 and the advantage is UNDEFINED.
# NEVER set this below 2.
NUM_GENERATIONS = 6  # drop to 4 on OOM

TIERS: dict[str, dict[str, Any]] = {
    "smoke": dict(max_steps=10, save_steps=5, num_generations=4),
    "full": dict(max_steps=300, save_steps=25, num_generations=6),
}

# --- Optimizer (known-good GRPO values) ---
LEARNING_RATE = 5e-6
OPTIM = "paged_adamw_8bit"
MAX_GRAD_NORM = 0.1
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

# --- Database ---
DB_PATH = "outputs/ecommerce.db"  # gitignored; rebuild with scripts/build_db.py
DB_SEED = 42  # deterministic seed — gold SQL results depend on it

# --- Environment (LangGraph loop) ---
MAX_TURNS = 6  # hard episode cap; also terminate on repeated identical tool calls
QUERY_ROW_CAP = 100  # LIMIT injected into every run_query
QUERY_TIMEOUT_S = 2.0

# --- Rewards ---
# total = 1.0 * execution_match + shaping; shaping capped at 0.3 and paid
# ONLY when execution_match == 1.0 (see rewards/hacking_guards.py).
EXECUTION_MATCH_WEIGHT = 1.0
FORMAT_REWARD = 0.1  # final answer inside <answer>...</answer>
EFFICIENCY_REWARD_MAX = 0.1  # 0.1 * (max_turns - turns_used) / max_turns
SCHEMA_DISCIPLINE_REWARD = 0.1  # describe_table before querying that table
SHAPING_CAP = 0.3

# --- Tasks ---
N_TASKS = 150
N_TRAIN = 120
N_VAL = 30
TASK_GEN_MODEL = "gpt-4o-mini"

# --- Baseline ---
BASELINE_MODEL = "gpt-4o-mini"

# --- Safety (see scripts/preflight.py and train/thermal_guard.py) ---
PREFLIGHT_MIN_FREE_VRAM_MB = 7168  # 7.0 GB
PREFLIGHT_MAX_IDLE_TEMP_C = 70
THERMAL_HALT_TEMP_C = 85  # checkpoint + halt if exceeded continuously...
THERMAL_HALT_SUSTAINED_S = 60  # ...for this long
THERMAL_POLL_INTERVAL_S = 10
