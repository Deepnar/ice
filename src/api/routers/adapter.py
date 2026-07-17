"""Shared router-adapter helpers (E0).

Routers are thin: parse → service → format. This module owns the one piece
of shared adapter logic — translating domain errors to HTTP. It may import
FastAPI; the services never do (grep-gated).
"""
from contextlib import contextmanager

from fastapi import HTTPException

from src.services.errors import ConflictError, NotFoundError, ValidationError


@contextmanager
def service_errors():
    """Translate service-layer domain errors to HTTPException."""
    try:
        yield
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
