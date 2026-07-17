"""Domain errors for the E0 service layer.

Services raise these; each adapter translates them to its own surface
(REST → 404/400/409 responses, MCP → tool error content, C11 chat commands
→ inline reply). Services themselves know nothing about HTTP.
"""


class ServiceError(Exception):
    """Base class for service-layer failures."""


class NotFoundError(ServiceError):
    """The addressed row/entity/model does not exist."""


class ValidationError(ServiceError):
    """The caller's input is outside the domain (bad slot name, bad action)."""


class ConflictError(ServiceError):
    """Concurrent-mutation conflict (e.g. the slot-initialization race)."""
