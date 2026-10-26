"""Every history Z2 (and later FINAL) runs, in one place.

Z2 D13: anything Z2 hard-codes to B/C is a parameter. A new history is an entry here,
not an edit across the tools. One history is one conversation replayed into one store;
a multi-chat group (FINAL candidates, Z3 D1) needs multi-conversation replay first.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFLINE = "logs/v3-offline-input-plan-2026-10-10-r4"


@dataclass(frozen=True)
class History:
    key: str
    slug: str                      # the conversation replayed (seed_v3 id)
    originals: str                 # role-separated originals in turn order (repo-relative)
    turns: int
    import_rows: bool              # replay rows come from `originals`, not seed_v3's corpus
    target_occurrences: int        # question occurrences before the Z2/FINAL split
    asks_per_family: float         # first ask + repeats, measured per history
    planned_asks: int              # the meter's planning number until the asks exist
    ledger: str = ""
    changes: str = ""
    asks: str = ""

    def path(self, field: str) -> Path:
        return ROOT / getattr(self, field)


HISTORIES = {
    "B": History("B", "bb558b5f", f"{OFFLINE}/final-originals-B.json", 1119, False, 500, 2.3, 345,
                 ledger="logs/z2-ledger-detail-2026-10-18-r2/B/ledger.json",
                 changes="logs/z2-questions-final/B/changes.json",
                 asks="logs/z2-questions-final/B/z2-asks.json"),
    # C's short history (325 turns, 13 checkpoints) gave 110 asks from 73 families = 1.5
    # (2026-10-19); B keeps the planned 2.3.
    "C": History("C", "a77c15cf", f"{OFFLINE}/final-originals-C.json", 325, True, 200, 1.5, 133,
                 ledger="logs/z2-ledger-detail-2026-10-18-r2/C/ledger.json",
                 changes="logs/z2-questions-final/C/changes.json",
                 asks="logs/z2-questions-final/C/z2-asks.json"),
}

# FINAL private histories (FINAL_experiments.md §2.1, 2026-10-26). Originals from
# experiments/final/private_originals.py; ledger, changes and asks are produced by Z2's
# S1-S6 chain into logs/final-questions/<key>/ (all of each history's families are FINAL's:
# no 60/40 split). Targets follow B's ~0.45 occurrences per pair; short histories ask
# 1.5 times per family, as C did.
FINAL_INPUTS = "logs/final-private-inputs"
FINAL_PRIVATE = {
    key: History(key, prefix, f"{FINAL_INPUTS}/final-originals-{key}.json", pairs, True,
                 round(pairs * 0.45), 1.5, round(pairs * 0.45 / 1.5),
                 ledger=f"logs/final-questions/{key}/ledger.json",
                 changes=f"logs/final-questions/{key}/changes.json",
                 asks=f"logs/final-questions/{key}/asks.json")
    for key, prefix, pairs in (("A", "633e26f8", 290), ("D", "ecc64aab", 251), ("SW", "bd94e64b", 128),
                               ("LAP", "942699a4", 101))
}
HISTORIES.update(FINAL_PRIVATE)
