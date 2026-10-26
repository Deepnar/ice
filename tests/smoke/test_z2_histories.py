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
    assert {k: R.HISTORY_SLUG[k] for k in "BC"} == {"B": "bb558b5f", "C": "a77c15cf"}
    assert R.EXTRA_HISTORIES["C"] == (H.HISTORIES["C"].originals, 325) and "B" not in R.EXTRA_HISTORIES
    assert {k: Q.TARGET_OCCURRENCES[k] for k in "BC"} == {"B": 500, "C": 200}
    assert {k: Q.ASKS_PER_FAMILY[k] for k in "BC"} == {"B": 2.3, "C": 1.5}


def test_final_private_histories_are_registered_for_import_replay():
    for key, pairs in (("A", 290), ("D", 251), ("SW", 128), ("LAP", 101)):
        h = H.HISTORIES[key]
        assert h.import_rows and h.turns == pairs and h.asks_per_family == 1.5
        assert R.EXTRA_HISTORIES[key] == (h.originals, pairs)
