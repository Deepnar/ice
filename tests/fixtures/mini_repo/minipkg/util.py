"""Utilities built on core."""
from minipkg.core import greet

import minipkg.config


def shout(name):
    """Uppercase greeting (resolved import call)."""
    return greet(name).upper()


def mystery():
    """Calls a bare name defined only in core (heuristic edge)."""
    return helper()  # noqa: F821 — deliberate: heuristic resolution fixture


def configured():
    """Attribute call on an imported project module (resolved)."""
    return minipkg.config.Settings()
