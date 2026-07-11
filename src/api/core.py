"""Core factory (C7 D10) — the E7 seam.

Builds the process's ICE core: the DB session factory plus the in-process
maintenance runtime. The FastAPI lifespan calls ``create_core()``; E7's
headless ``ice-mcp`` boot will call the same factory without HTTP. Neither
this module nor ``runtime.py`` may import FastAPI or ``src.api.main`` —
that coupling is exactly what the split exists to remove.
"""

from dataclasses import dataclass
from typing import Callable

from src.api.db import SessionLocal
from src.workers.runtime import MaintenanceRuntime


@dataclass
class ICECore:
    db_factory: Callable
    runtime: MaintenanceRuntime

    async def stop(self) -> None:
        await self.runtime.stop()


def create_core() -> ICECore:
    """Build and start the core. Requires a running event loop (the runtime
    spawns its tick task on it)."""
    runtime = MaintenanceRuntime()
    runtime.start(SessionLocal)
    return ICECore(db_factory=SessionLocal, runtime=runtime)
