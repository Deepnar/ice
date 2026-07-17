"""Model-registry service (E0) — file-backed, guarded by an fcntl lock.

The registry JSON is shared mutable state between the app UI and ice-mcp
(`ice_control registry_edit`); every load-modify-save runs inside one
exclusive lock (last-writer-wins *inside* the lock, spec §4). Reads take the
same lock briefly so a reader can never observe a half-written file.
"""
import fcntl
from contextlib import contextmanager

import src.model_registry.registry as registry
from src.services.errors import NotFoundError

# The four caller-editable fields (same set the old router accepted).
_EDITABLE_FIELDS = ("topic_tags", "intent_tags", "confirmed", "base_url")


@contextmanager
def _registry_lock():
    # Sidecar lockfile (save_registry truncate-writes the data file, so
    # locking the data fd itself would not span the load-modify-save).
    # REGISTRY_PATH read dynamically — tests repoint it.
    with open(registry.REGISTRY_PATH + ".lock", "w") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def get_registry() -> dict:
    with _registry_lock():
        return registry.load_registry()


def refresh_registry() -> dict:
    with _registry_lock():
        return registry.populate_from_ollama()


def update_model(model_name: str, data: dict) -> dict:
    with _registry_lock():
        reg = registry.load_registry()
        if model_name not in reg["models"]:
            raise NotFoundError("Model not found")
        entry = reg["models"][model_name]
        for field in _EDITABLE_FIELDS:
            if field in data:
                entry[field] = data[field]
        registry.save_registry(reg)
    return {"status": "updated"}


def delete_model(model_name: str) -> dict:
    with _registry_lock():
        reg = registry.load_registry()
        if model_name not in reg["models"]:
            raise NotFoundError("Model not found")
        del reg["models"][model_name]
        registry.save_registry(reg)
    return {"status": "deleted"}
