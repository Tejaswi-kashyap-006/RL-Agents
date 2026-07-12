"""VRAM/temperature probes via nvidia-smi.

Used by scripts/preflight.py (safety gate) and train/thermal_guard.py
(watchdog). Deliberately dependency-free: nvidia-smi is the source of
truth, and this must work before any Python deps are installed.
"""

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSnapshot:
    """One point-in-time reading of GPU 0."""

    name: str
    free_vram_mb: int
    total_vram_mb: int
    temp_c: int


def query_gpu(timeout_s: float = 10.0) -> GpuSnapshot | None:
    """Return a snapshot of GPU 0, or None if nvidia-smi is missing or fails.

    A None return means CUDA is effectively unavailable — callers must
    treat it as a hard failure, not silently continue.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        name, free_mb, total_mb, temp = (f.strip() for f in out.stdout.strip().split(","))
        return GpuSnapshot(
            name=name,
            free_vram_mb=int(free_mb),
            total_vram_mb=int(total_mb),
            temp_c=int(temp),
        )
    except ValueError:
        return None
