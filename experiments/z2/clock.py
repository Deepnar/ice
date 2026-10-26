"""Z2 declared session clock (spec Z2_EVALUATION D3).

B's and C's timestamps are synthetic imports compressed into a few days, so
decay and the maintenance cadence cannot act plausibly on them. Z2 replays them
on a declared clock instead: sessions of 25 consecutive pairs, 3 minutes
between turns inside a session, and between sessions a deterministic seeded gap
of 1, 2 or 3 days (measured from the previous session's last turn). The first
turn keeps the history's own first timestamp as the anchor.

Authentic timestamps are never replaced: a history whose provenance is
``original`` is returned unchanged. The table is written before the replay and
is a USER-REQUIRED review (D3).

    uv run --no-sync python experiments/z2/clock.py --history B --out logs/z2-clock/B.json
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-declared-clock-1"
SESSION_PAIRS = 25
TURN_GAP = timedelta(minutes=3)
GAP_DAYS = (1, 2, 3)
INPUTS = ROOT / "logs/v3-offline-input-plan-2026-10-10-r4"
SYNTHETIC = "synthetic_raw_import"


def session_gap(history: str, session: int) -> timedelta:
    """Gap before *session* (1-based, > 0). Hash-seeded, so no RNG state or
    library version can change it."""
    seed = hashlib.sha256(f"{POLICY}|{history}|{session}".encode()).digest()
    return timedelta(days=GAP_DAYS[seed[0] % len(GAP_DAYS)])


def declared_times(history: str, count: int, anchor: datetime) -> list:
    """Clock time for pairs 1..count."""
    if anchor.tzinfo is None:
        raise ValueError("clock anchor lacks a timezone")
    times, current = [], anchor
    for index in range(count):
        session, position = divmod(index, SESSION_PAIRS)
        if index:
            current += session_gap(history, session) if position == 0 else TURN_GAP
        times.append(current)
    return times


def table(history: str, rows: list) -> dict:
    """The clock table for one history's original pairs (turn order as given)."""
    provenance = {r["timestamp_provenance"] for r in rows}
    if len(provenance) != 1:
        raise ValueError(f"{history}: mixed timestamp provenance {sorted(provenance)}")
    (kind,) = provenance
    if [r["turn_number"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{history}: turns are not 1..n in order")
    originals = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in rows]
    if kind == SYNTHETIC:
        times = declared_times(history, len(rows), originals[0])
        basis = "declared"
    elif kind == "original":
        times, basis = originals, "original_unchanged"
    else:
        raise ValueError(f"{history}: unknown timestamp provenance {kind!r}")
    sessions = []
    for start in range(0, len(rows), SESSION_PAIRS):
        span = times[start:start + SESSION_PAIRS]
        sessions.append(dict(session=len(sessions) + 1, first_turn=start + 1,
                             last_turn=start + len(span), start=span[0].isoformat(),
                             end=span[-1].isoformat()))
    return dict(policy=POLICY, history=history, basis=basis, source_provenance=kind,
                session_pairs=SESSION_PAIRS, turn_gap_seconds=int(TURN_GAP.total_seconds()),
                gap_days=list(GAP_DAYS), pairs=len(rows),
                span_days=round((times[-1] - times[0]).total_seconds() / 86400, 2),
                sessions=sessions, turn_times=[t.isoformat() for t in times])


def load_history(history: str) -> list:
    from experiments.z2.histories import HISTORIES
    return json.loads(HISTORIES[history].path("originals").read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    from experiments.z2.histories import HISTORIES
    parser.add_argument("--history", choices=sorted(HISTORIES), required=True)
    parser.add_argument("--out", type=Path, help="write the table (refuses to overwrite)")
    args = parser.parse_args()
    result = table(args.history, load_history(args.history))
    print(f"{args.history}: {result['basis']}, {result['pairs']} pairs, "
          f"{len(result['sessions'])} sessions over {result['span_days']} days")
    if args.out:
        with args.out.open("x") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
