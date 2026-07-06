# ICE Source Structure

Root: `/home/deepnar/Programs/ice/src`

```text
src/
├── api
│   ├── routers
│   │   ├── memory_slots.py
│   │   └── user_control.py
│   ├── config.py
│   ├── db.py
│   ├── main.py
│   └── prompt_assembler.py
├── classifier
│   ├── classifier.py
│   ├── dataset.py
│   ├── di3.py
│   ├── di3_config.py
│   ├── di3_logger.py
│   ├── di3_signals.py
│   ├── model.py
│   ├── ner_model.py
│   └── schemas.py
├── memory
│   └── models.py
├── model_registry
│   └── registry.py
├── retrieval
│   ├── configurable_orchestrator.py
│   ├── mera.py
│   ├── ner_utils.py
│   └── orchestrator.py
└── workers
    ├── batch_summarizer.py
    ├── bg_client_factory.py
    ├── celery_app.py
    ├── clustering.py
    ├── codex_decay.py
    ├── codex_extractor.py
    ├── codex_inject_watcher.py
    ├── compaction.py
    ├── decay.py
    ├── drop_zone.py
    ├── fine_tune.py
    ├── gpu_check.py
    ├── post_flight.py
    ├── procedural_decay.py
    ├── procedural_extractor.py
    ├── reflection.py
    └── sentinel_monitor.py
```
