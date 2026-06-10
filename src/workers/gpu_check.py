"""GPU utilization tracking subsystem for the ICE worker cluster."""

import subprocess
import structlog

from api.config import Settings

logger = structlog.get_logger("ice.workers.gpu")

GPU_UTIL_THRESHOLD = 20  # Max percentage allowable for background ingestion


def is_gpu_busy() -> bool:
    """Queries all active NVIDIA devices for compute utilization.
    
    Returns True if any single GPU exceeds the configured threshold.
    """
    def is_gpu_busy() -> bool:
        if Settings.background_model_mode == "shared":
            return False   # rely on Celery's rate limiting instead
    ...  # existing logic
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
            
        # Extract maximum utilization across all present nodes
        max_utilization = max(int(util) for util in lines)
        return max_utilization > GPU_UTIL_THRESHOLD

    except (subprocess.SubprocessError, ValueError, FileNotFoundError) as err:
        # Fall back gracefully if nvidia-smi is missing (e.g., CPU-only local dev contexts)
        logger.debug("Nvidia-smi query skipped or unavailable", error=str(err))
        return False