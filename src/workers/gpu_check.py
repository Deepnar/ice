"""GPU utilization tracking subsystem for the ICE worker cluster."""

import subprocess
import structlog
from src.api.config import settings
import redis
from datetime import datetime, timezone, timedelta

logger = structlog.get_logger("ice.workers.gpu")
GPU_UTIL_THRESHOLD = 20


def is_gpu_busy() -> bool:
    """Queries all active NVIDIA devices for compute utilization.

    Returns True if any single GPU exceeds the configured threshold.
    In shared mode, always returns False (background workers yield manually).
    """
    if settings.background_model_mode == "shared":
        return False

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if not lines:
            return False
        max_utilization = max(int(util) for util in lines)
        return max_utilization > GPU_UTIL_THRESHOLD
    except (subprocess.SubprocessError, ValueError, FileNotFoundError) as err:
        logger.debug("Nvidia-smi query skipped or unavailable", error=str(err))
        return False
    
def is_user_active(idle_threshold_seconds: int = 10) -> bool:
    """Return True if the user has chatted recently (shared mode should yield)."""
    try:
        # decode_responses=True: r.get() otherwise returns bytes, and
        # datetime.fromisoformat(bytes) raises TypeError — which the bare except
        # swallowed, making this silently always-False (G21 audit fix).
        r = redis.from_url(settings.redis_url, decode_responses=True)
        last = r.get("ice:last_chat_completed")
        r.close()
        if last:
            last_time = datetime.fromisoformat(last)
            return (datetime.now(timezone.utc) - last_time).total_seconds() < idle_threshold_seconds
    except Exception as err:
        logger.debug("is_user_active check failed", error=str(err))
    return False