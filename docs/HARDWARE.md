# Hardware — 8 GB budget, OOM ladder, thermal guidance

Target: laptop RTX 4060, 8 GB VRAM, 16 GB system RAM, Windows + WSL2.

## Memory budget — do not exceed

| Component | Budget |
|---|---|
| Policy model (Qwen2.5-1.5B-Instruct, QLoRA 4-bit) | ~2.0 GB |
| vLLM rollout engine (`gpu_memory_utilization=0.55`) | ~3.5 GB |
| LoRA grads + optimizer states + activations | ~1.5 GB |
| Headroom for Windows display + spikes | ~1.0 GB |
| **Total** | **~8.0 GB — at the limit** |

## OOM ladder

Apply in this order, one step at a time, re-running the smoke tier between steps:

1. `NUM_GENERATIONS` 6 → 4 (never below 2 — the group baseline degenerates)
2. reduce `MAX_SEQ_LENGTH` (1536 → 1280 → 1024)
3. reduce `GPU_MEMORY_UTILIZATION` (0.55 → 0.50 → 0.45)
4. reduce `LORA_RANK` (16 → 8)

All in `src/agent_sql/config.py`.

## Thermal guidance

The card cannot be damaged — firmware throttles it — but a laptop chassis under
multi-hour sustained load is loud, hot, and hard on everything around the GPU. The
built-in guards: preflight requires < 70 °C at idle; the training watchdog checkpoints
and halts if temp exceeds 85 °C continuously for 60 s.

Practical measures: hard surface (not a bed/lap), elevate the rear, max fan profile,
**never on battery**.

## Optional power cap (recommended for long runs)

`nvidia-smi -pl <watts>` caps board power, cutting heat substantially for a modest speed
cost. It is **not** enforced by any code in this repo — it is a manual, admin-level
choice. It resets on reboot.

Read your card's default and allowed range first:

```
nvidia-smi -q -d POWER
```

Look for `Default Power Limit`, `Min Power Limit`, and `Max Power Limit`. Laptop RTX
4060s ship anywhere from ~60 W to ~115 W depending on the OEM, so read yours rather than
assuming. A sensible cap for a long run is 75–85 % of the default — e.g. if the default
is 100 W:

```
nvidia-smi -pl 80    # run as Administrator (Windows) / with sudo effect via Windows side
```

Set it from the **Windows** side (the Windows driver owns the hardware; a cap set there
applies to WSL workloads too). If you get "Changing power management limit is not
supported", your OEM has locked it — rely on the thermal watchdog instead.
