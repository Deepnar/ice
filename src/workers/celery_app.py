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
        "src.workers.batch_summarizer",
        "src.workers.cluster_merge",
    ],
)

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"

app.conf.beat_schedule = {
    # ── Lightweight, no GPU, frequent ──
    'cluster-turns': {
        'task': 'src.workers.clustering.cluster_turns',
        'schedule': 1800.0,          # every 30 min
    },
    'monitor-sentinels': {
        'task': 'src.workers.sentinel_monitor.monitor_sentinels',
        'schedule': 1800.0,
    },

    # ── Lightweight decay (tiny multipliers, harmless to run often) ──
    'decay-episodic': {
        'task': 'src.workers.decay.apply_decay',
        'schedule': 5400.0,          # every 1.5 hours
    },
    'decay-codex': {
        'task': 'src.workers.codex_decay.decay_codex_edges',
        'schedule': 5400.0,
    },
    'decay-procedural': {
        'task': 'src.workers.procedural_decay.decay_procedural_patterns',
        'schedule': 5400.0,
    },

    # ── GPU‑heavy, moderate frequency ──
    'reflection': {
        'task': 'src.workers.reflection.run_reflection',
        'schedule': 7200.0,          # every 2 hours
    },
    'batch-summarize': {
        'task': 'src.workers.batch_summarizer.batch_summarize',
        'schedule': 7200.0,
    },

    # ── Extremely rare ──
    'fine-tune-weekly': {
        'task': 'src.workers.fine_tune.fine_tune_classifier',
        'schedule': crontab(hour=4, minute=0, day_of_week=1),
    },
}