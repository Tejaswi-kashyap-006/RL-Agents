#!/usr/bin/env bash
# Live GPU temp/VRAM monitor. Run in a second terminal during training.
# Usage: bash scripts/watch_gpu.sh [interval_seconds]
set -euo pipefail
interval="${1:-5}"
echo "temp_C, mem_used_MB, mem_total_MB, util_%   (Ctrl-C to stop)"
while true; do
    nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
    sleep "$interval"
done
