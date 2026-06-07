from celery import Celery
from src.api.config import settings

app = Celery(
    "ice_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "src.workers.post_flight",
        # Future workers will be added here:
        # "src.workers.codex_extractor",
        # "src.workers.procedural_extractor",
        # "src.workers.reflection",
        # "src.workers.decay",
        # "src.workers.sentinel_monitor",
    ],
)

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"