"""Z2 D13: one registry of histories; the tools derive their per-history values from
it, and B/C keep exactly the values Z2 ran with."""

from experiments.z2 import histories as H
from experiments.z2 import questions as Q
from experiments.z2 import replay as R


def test_registry_holds_the_values_z2_ran_with():
    b, c = H.HISTORIES["B"], H.HISTORIES["C"]
    assert (b.slug, b.turns, b.import_rows, b.target_occurrences, b.asks_per_family) == ("bb558b5f", 1119, False, 500, 2.3)
    assert (c.slug, c.turns, c.import_rows, c.target_occurrences, c.asks_per_family) == ("a77c15cf", 325, True, 200, 1.5)


def test_tools_derive_from_the_registry():
    assert R.HISTORY_SLUG == {"B": "bb558b5f", "C": "a77c15cf"}
    assert R.EXTRA_HISTORIES == {"C": (H.HISTORIES["C"].originals, 325)}
    assert Q.TARGET_OCCURRENCES == {"B": 500, "C": 200} and Q.ASKS_PER_FAMILY == {"B": 2.3, "C": 1.5}
