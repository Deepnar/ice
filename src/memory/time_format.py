"""Render stored clock values without inventing their origin or timezone."""

from datetime import datetime, timezone


def format_time(value: datetime | None) -> str:
    if value is None:
        return "unknown"
    if value.tzinfo is None or value.utcoffset() is None:
        return value.isoformat(timespec="seconds") + " (timezone unknown)"
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def recorded_stamp(value, provenance=None) -> str:
    if value is None:
        return ""
    if provenance == "original":
        label = "source recorded"
    elif provenance == "synthetic_raw_import":
        label = "imported; source time unknown"
    else:
        label = "recorded; provenance unknown"
    return f"[{label}: {format_time(value)}] "
