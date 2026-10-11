"""Scoped native logged-exception receipts, without changing worker policy.

Unlogged swallowed exceptions are not observable here. Serial isolated research
execution only: native logger bindings are restored on every exit.
"""

import sys
from contextlib import contextmanager


class ObservedLogger:
    def __init__(self, logger, faults, context, module):
        self.logger, self.faults, self.context, self.module = logger, faults, context, module

    def __getattr__(self, name):
        method = getattr(self.logger, name)
        if name in {"bind", "new", "unbind", "try_unbind"}:
            def bind(*args, **kwargs):
                return ObservedLogger(method(*args, **kwargs), self.faults, self.context, self.module)
            return bind
        if name not in {"warning", "warn", "error", "exception", "critical"}:
            return method

        def emit(*args, **kwargs):
            exc = sys.exc_info()[1]
            if exc is not None:
                event = args[0] if args and isinstance(args[0], str) else name
                self.faults.record(exc, {**self.context,
                    "stage": f"logged_caught:{self.module}:{event}"})
            return method(*args, **kwargs)
        return emit


@contextmanager
def observe(faults, context):
    """Call after loading native modules; do not infer exhaustive capture."""
    changed = []
    try:
        for name, module in list(sys.modules.items()):
            if name.startswith("src.") and hasattr(module, "logger"):
                logger = module.logger
                if isinstance(logger, ObservedLogger):
                    raise ValueError("caught-fault observers must not overlap")
                changed.append((module, logger))
                module.logger = ObservedLogger(logger, faults, context, name)
        yield
    finally:
        for module, logger in reversed(changed):
            module.logger = logger
