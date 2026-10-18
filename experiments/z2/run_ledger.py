"""Z2 S1 + S2 driver: both writers extract each history (in parallel), then merge.

    uv run --no-sync python experiments/z2/run_ledger.py --out logs/z2-ledger-<date>-r1 --history B C

Each history gets `out/<history>/` with the two `ledger-raw-*.jsonl`, the merge's
`merge-raw.jsonl`, `ledger.json` and `review-queue.json`. Every step is resumable
(window lines and model calls are durable and identity-checked): rerunning the
same command continues where it stopped. A history whose extraction fails is
reported and skipped; the others still run. Exit status is non-zero if any failed.
"""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import ledger, ledger_merge
from experiments.z2.profiles import TIE_BREAK, WRITERS, generator, metadata


def extract(history, directory):
    originals = ledger.load_history(history)
    plan = [w["new"] for w in ledger.windows(originals)]
    for writer in WRITERS:          # a saved run planned with other windows cannot be merged
        path = directory / f"ledger-raw-{writer}.jsonl"
        saved = [json.loads(line)["new"] for line in path.open()] if path.exists() else []
        if saved != plan[:len(saved)]:
            raise RuntimeError(f"{path.name}: saved windows differ from the plan; move it aside")

    def one(writer):
        started = time.monotonic()
        ledger.run_writer(history, writer, directory)
        return writer, round(time.monotonic() - started)

    with ThreadPoolExecutor(max_workers=len(WRITERS)) as pool:
        return dict(pool.map(one, WRITERS))


def merge(history, directory):
    roles = set(WRITERS) | {TIE_BREAK}
    models = {role: (generator(role).generate, metadata(role)) for role in roles}
    result, queue = ledger_merge.merge(history, directory, directory, writers=WRITERS,
                                       tie_breaker=TIE_BREAK, models=models)
    return result["stats"], len(queue)


def load_tokenizer():
    """Load the token counter before any writer thread starts. Its lazy loader is
    not thread-safe: a thread arriving during the first load gets None and counts
    with chars/4, silently (C, 2026-10-18: MiMo planned 1-16 where the real
    count gives 1-14, so the two writers' windows differed)."""
    from src.memory import tokens
    tokens.count("warm")
    if tokens._get_tokenizer() is None:
        raise RuntimeError("tokenizer did not load; window plans would use an estimate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="private run root under logs/")
    parser.add_argument("--history", nargs="+", required=True, help="B, C or dev-<slug>")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("ledger data is private: put it under logs/")
    load_tokenizer()
    failed = []
    for history in args.history:
        directory = args.out / history
        directory.mkdir(parents=True, exist_ok=True)
        try:
            seconds = extract(history, directory)
            print(f"[{history}] extraction complete: {seconds}", flush=True)
            if not args.extract_only:
                stats, queued = merge(history, directory)
                print(f"[{history}] merge complete: {json.dumps(stats)} queue {queued}", flush=True)
        except Exception:                               # noqa: BLE001 — report, go on
            failed.append(history)
            print(f"[{history}] FAILED\n{traceback.format_exc()}", flush=True)
    print(f"DONE; failed: {failed or 'none'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
