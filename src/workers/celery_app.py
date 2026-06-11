from celery import Celery
from celery.schedules import crontab
from src.api.config import settings

app = Celery(
    "ice_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "src.workers.post_flight",
        "src.workers.codex_extractor",
        "src.workers.compaction",
        "src.workers.procedural_extractor",
        "src.workers.decay",
        "src.workers.reflection",
        "src.workers.sentinel_monitor",
        "src.workers.clustering",
        "src.workers.fine_tune",
        "src.workers.codex_decay",
        "src.workers.procedural_decay",
    ],
)

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"

app.conf.beat_schedule = {
    'apply-decay-daily': {
        'task': 'src.workers.decay.apply_decay',
        'schedule': crontab(hour=3, minute=0),
    },
    'codex-decay-daily': {
        'task': 'src.workers.codex_decay.decay_codex_edges',
        'schedule': crontab(hour=3, minute=30),
    },
    'procedural-decay-daily': {
        'task': 'src.workers.procedural_decay.decay_procedural_patterns',
        'schedule': crontab(hour=4, minute=30),
    },
    'cluster-turns-daily': {
        'task': 'src.workers.clustering.cluster_turns',
        'schedule': crontab(hour=4, minute=0),
    },
    'monitor-sentinels': {
        'task': 'src.workers.sentinel_monitor.monitor_sentinels',
        'schedule': crontab(minute='*/30'),
    },
    'reflection-daily': {
        'task': 'src.workers.reflection.run_reflection',
        'schedule': crontab(hour=5, minute=0),
    },
    'fine-tune-weekly': {
        'task': 'src.workers.fine_tune.fine_tune_classifier',
        'schedule': crontab(hour=4, minute=0, day_of_week=1),
    },
}