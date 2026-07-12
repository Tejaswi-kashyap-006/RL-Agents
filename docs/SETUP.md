# Setup — WSL2 + CUDA + deps

vLLM does not run natively on Windows, and GRPO depends on vLLM for rollout generation,
so **training runs inside WSL2 (Ubuntu 22.04) with NVIDIA GPU passthrough**. This is the
single biggest setup trap in the project — follow the steps in order.

Everything up to training (DB, tools, environment, tasks, rewards, tests, evaluation of
the GPT-4o-mini baseline) also works on Windows-side Python. Only `scripts/train.py`
needs WSL.

## 1. Enable WSL2 and install Ubuntu 22.04

In an **admin** PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot if prompted, create the Linux user, then confirm:

```powershell
wsl --status   # should say: Default Version: 2
```

## 2. NVIDIA driver — Windows side ONLY

Install/update the normal Windows NVIDIA driver (GeForce/Studio driver from nvidia.com).

**Do NOT install any NVIDIA driver inside WSL.** GPU passthrough is handled by the
Windows driver; a Linux driver inside WSL will break it.

Verify passthrough from inside Ubuntu:

```bash
nvidia-smi   # must show the RTX 4060 Laptop GPU
```

If `nvidia-smi` fails inside WSL, stop and fix this before anything else.

## 3. Cap WSL memory — `~/.wslconfig` (Windows side)

On a 16 GB machine, WSL will happily eat all of it and thrash the host. Create
`C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=8GB
```

This leaves 4 GB for Windows. Apply with `wsl --shutdown`, then reopen Ubuntu.

## 4. CUDA toolkit, uv, and deps (inside WSL)

```bash
# CUDA toolkit for WSL (NO driver — toolkit only). Follow NVIDIA's
# "CUDA on WSL" instructions for the wsl-ubuntu 12.x package, e.g.:
sudo apt-key del 7fa2af80  # remove old key if present, per NVIDIA docs
# ...then the cuda-toolkit-12-* install steps from
# https://developer.nvidia.com/cuda-downloads (Linux → x86_64 → WSL-Ubuntu)

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# project deps
cd /mnt/c/Users/<you>/Desktop/development/rl_agents   # or clone into the WSL filesystem (faster I/O)
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"

# training stack — LAST, and only after nvidia-smi works in WSL.
# These pins are the fragile part; if versions have drifted, install unsloth
# first and let it pin torch/trl/vllm itself:
uv pip install unsloth vllm trl
```

Note: keep the WSL `.venv` separate from any Windows-side `.venv` — they are not
interchangeable. If you work from `/mnt/c/...`, create the WSL venv somewhere inside the
WSL filesystem (e.g. `~/venvs/grpo`) to avoid both slowness and collisions.

## 5. Preflight

```bash
python scripts/preflight.py
```

All checks must pass before any training run. No exceptions.

## Escape hatches — when the install fights back

WSL + Unsloth installs are notoriously fragile; bitsandbytes/numpy/torch version
conflicts are common. If you have burned more than an hour or two on the environment:

**(a) Unsloth's official Docker image.** Supports Windows/WSL/Linux; ships with the full
pinned stack. Requires Docker Desktop with the WSL2 backend and GPU support enabled:

```bash
docker run --gpus all -it -v $(pwd):/workspace unsloth/unsloth
```

**(b) Train in Colab, run inference/eval locally.** A T4 (16 GB) on the free tier fits
this config comfortably. Sync checkpoints back and run `scripts/evaluate.py` locally —
evaluation does not need vLLM.
