"""Clock substitution preserves ordinary datetime parsing and type checks."""
from datetime import datetime, timezone

import pytest

from scripts.z1.historical_clock import source_datetime


def test_source_clock_preserves_real_datetimes_and_timezone():
    source = datetime.fromisoformat("2025-01-01T03:00:00+05:30")
    clock = source_datetime(source)
    assert clock.now(timezone.utc).isoformat() == "2024-12-31T21:30:00+00:00"
    assert clock.utcnow() == datetime(2024, 12, 31, 21, 30)
    assert isinstance(source, clock)
    assert clock.fromisoformat("2025-01-02T00:00:00+00:00").day == 2
    with pytest.raises(ValueError, match="timezone-aware"):
        source_datetime(datetime(2025, 1, 1))
