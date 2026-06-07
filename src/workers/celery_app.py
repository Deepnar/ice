from celery import Celery
from src.api.config import settings

app = Celery(
    "ice_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "src.workers.post_flight",
        "src.workers.codex_extractor",   # ← add this
        "src.workers.compaction",        # ← add this
    ],
)

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"