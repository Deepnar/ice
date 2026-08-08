"""GPU utilization gate — dedicated mode only (C7 D7/G4).

In shared mode the background model IS the chat model, so hardware polling is
meaningless there: idleness is in-process truth (the maintenance runtime's
generation-inflight flag + last-activity recency, see runtime.is_idle()).
nvidia-smi survives solely for dedicated mode, where a separate bg server
shares the card with the chat model: threshold 70 (20 stalled bg work whenever
the desktop compositor blipped), result cached 10s.

The old is_user_active() redis check died with the broker — its semantics
live in MaintenanceRuntime (user_active_threshold_seconds = 90; the 10s
window it hardcoded was uselessly tight).
"""

import subprocess
import time

import structlog

from src.api.config import settings

logger = structlog.get_logger("ice.workers.gpu")

_cache: tuple = (0.0, False)   # (monotonic_stamp, busy)


def is_gpu_busy() -> bool:
    """Dedicated mode: True when any NVIDIA device exceeds the threshold
    (cached 10s). Shared mode: always False — the runtime gates instead."""
    global _cache
    if settings.background_model_mode == "shared":
        return False

    stamp, busy = _cache
    now = time.monotonic()
    if now - stamp < settings.gpu_check_cache_seconds:
        return busy

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        busy = bool(lines) and max(int(util) for util in lines) > settings.gpu_util_threshold
    except (subprocess.SubprocessError, ValueError, FileNotFoundError) as err:
        logger.debug("Nvidia-smi query skipped or unavailable", error=str(err))
        busy = False

    _cache = (now, busy)
    return busy
