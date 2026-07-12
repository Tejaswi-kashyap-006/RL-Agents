"""SAFETY GATE — run before any training script. No training without a full pass.

Checks, in order:
  1. CUDA visible        — nvidia-smi responds (proxy: no torch dep required here)
  2. Free VRAM >= 7.0 GB — the 8 GB budget has ~1 GB headroom; less means OOM
  3. GPU temp < 70 C     — must start cool; a warm start eats the thermal margin
  4. AC power connected  — NEVER train on battery

Exit code 0 only if every check passes. Any failure aborts with a clear
message. Dependency-free by design: this must run before deps are installed.
"""

import subprocess
import sys
from pathlib import Path

# Allow running as `python scripts/preflight.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_sql.config import PREFLIGHT_MAX_IDLE_TEMP_C, PREFLIGHT_MIN_FREE_VRAM_MB  # noqa: E402
from agent_sql.utils.gpu import GpuSnapshot, query_gpu  # noqa: E402


def check_ac_power() -> tuple[bool, str]:
    """Return (on_ac, detail). Unknown power state counts as a FAILURE —
    training on battery is the one mistake this gate exists to prevent."""
    if sys.platform == "win32":
        return _ac_power_via_powershell("powershell")
    # Linux: real hardware exposes a Mains supply in sysfs; WSL2 usually
    # does not, so fall back to querying Windows through interop.
    supply_root = Path("/sys/class/power_supply")
    if supply_root.is_dir():
        for supply in supply_root.iterdir():
            try:
                if (supply / "type").read_text().strip() == "Mains":
                    online = (supply / "online").read_text().strip()
                    return online == "1", f"sysfs {supply.name}: online={online}"
            except OSError:
                continue
    return _ac_power_via_powershell("powershell.exe")  # WSL interop


def _ac_power_via_powershell(exe: str) -> tuple[bool, str]:
    cmd = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "[System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus"
    )
    try:
        out = subprocess.run(
            [exe, "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "could not query Windows power status"
    status = out.stdout.strip()
    if status == "Online":
        return True, "PowerLineStatus=Online"
    return False, f"PowerLineStatus={status or 'unknown'}"


def main() -> int:
    """Run all checks; print an honest report; return 0 only on full pass."""
    failures: list[str] = []

    print("=== preflight: safety gate ===")

    gpu: GpuSnapshot | None = query_gpu()
    if gpu is None:
        print("[FAIL] CUDA visible: nvidia-smi missing or unresponsive")
        failures.append(
            "No GPU visible. Check the NVIDIA driver (Windows-side only if in WSL) "
            "and that `nvidia-smi` works in this shell."
        )
    else:
        print(f"[PASS] CUDA visible: {gpu.name} ({gpu.total_vram_mb} MB total)")

        if gpu.free_vram_mb >= PREFLIGHT_MIN_FREE_VRAM_MB:
            print(f"[PASS] Free VRAM: {gpu.free_vram_mb} MB >= {PREFLIGHT_MIN_FREE_VRAM_MB} MB")
        else:
            print(f"[FAIL] Free VRAM: {gpu.free_vram_mb} MB < {PREFLIGHT_MIN_FREE_VRAM_MB} MB")
            failures.append(
                f"Only {gpu.free_vram_mb} MB VRAM free. Close GPU-hungry apps "
                "(browsers, games, other CUDA processes) and retry."
            )

        if gpu.temp_c < PREFLIGHT_MAX_IDLE_TEMP_C:
            print(f"[PASS] GPU temp: {gpu.temp_c} C < {PREFLIGHT_MAX_IDLE_TEMP_C} C")
        else:
            print(f"[FAIL] GPU temp: {gpu.temp_c} C >= {PREFLIGHT_MAX_IDLE_TEMP_C} C")
            failures.append(
                f"GPU is at {gpu.temp_c} C before training even starts. "
                "Let the machine cool down; check airflow."
            )

    on_ac, detail = check_ac_power()
    if on_ac:
        print(f"[PASS] AC power: {detail}")
    else:
        print(f"[FAIL] AC power: {detail}")
        failures.append("Not confirmed on AC power. NEVER train on battery. Plug in and retry.")

    if failures:
        print("\npreflight FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\npreflight PASSED - clear to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
