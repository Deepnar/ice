"""Core factory (C7 D10, extended by E0/E7) — the E7 seam.

Builds the process's ICE core: the DB session factory, the in-process
maintenance runtime, and (lazily) the classifier/embedder pair. The FastAPI
lifespan calls ``create_core()``; E7's headless ``ice-mcp`` boot calls the
same factory without HTTP. Neither this module nor ``runtime.py`` may import
FastAPI or ``src.api.main`` — that coupling is exactly what the split exists
to remove.

Runtime start is LEASE-CHECKED by default (E7 D6): if another process's
runtime holds a fresh ``runtime_lease``, this core's runtime starts in
standby — event jobs for this process still run, periodic maintenance stays
with the lease owner, and the standby promotes itself when the owner exits.
Same check both directions (app and ice-mcp).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from src.api.db import SessionLocal
from src.workers.runtime import MaintenanceRuntime, runtime_lease_fresh


@dataclass
class ICECore:
    db_factory: Callable
    runtime: Optional[MaintenanceRuntime]
    _classifier: Optional[object] = field(default=None, repr=False)

    @property
    def classifier(self):
        """The process's ONE classifier/embedder (G13). Lazy: a headless core
        skips the torch/embedder load until the first classify/encode — the
        app's lifespan touches this eagerly to keep boot-time loading."""
        if self._classifier is None:
            # Lazy import: torch + sentence-transformers only when needed.
            from src.api.config import settings
            from src.classifier.classifier import PyTorchClassifier
            self._classifier = PyTorchClassifier(
                model_path=settings.classifier_model_path,
                schema_path=settings.label_schema_path,
            )
        return self._classifier

    async def stop(self) -> None:
        global _active_core
        if self.runtime is not None:
            await self.runtime.stop()
        if _active_core is self:
            _active_core = None


# Module-level handle so services (retrieval_svc's context_for) reach the
# live classifier without a second model load — mirrors runtime.get_runtime().
_active_core: Optional[ICECore] = None


def get_core() -> Optional[ICECore]:
    return _active_core


def create_core(start_runtime: Optional[bool] = None) -> ICECore:
    """Build and start the core. Requires a running event loop (the runtime
    spawns its tick task on it).

    start_runtime: None (default) ⇒ lease-checked — full runtime when the
    lease is stale/absent, standby when another process owns it; True ⇒
    force-own (tests/recovery); False ⇒ no runtime at all (pure-read tools,
    test fixtures).
    """
    global _active_core
    if start_runtime is False:
        runtime = None
    else:
        standby = (start_runtime is None) and runtime_lease_fresh(SessionLocal)
        runtime = MaintenanceRuntime()
        runtime.start(SessionLocal, standby=standby)
    core = ICECore(db_factory=SessionLocal, runtime=runtime)
    _active_core = core
    return core
