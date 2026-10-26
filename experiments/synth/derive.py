"""FINAL synthetic set: regenerate every derived file from the sources, and pin them.

Sources (committed): the plans S1-S7, S9 (<ID>.json) and the question files (<ID>.questions.json).
Derived (not committed, regenerated): S8.json (S1's style twin), <ID>.occurrences.json,
<ID>.asks.json, <ID>.final-occurrences.json, <ID>.ledger.json. DERIVED.sha256 pins them: a
regeneration that comes out different is refused, so the repo stays small and FINAL's inputs
stay exactly the ones reviewed.

    uv run --no-sync python experiments/synth/derive.py           # regenerate + verify
    uv run --no-sync python experiments/synth/derive.py --pin     # (re)write the pins
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.synth import keys, phrase, plans  # noqa: E402

PINS = plans.PLANS / "DERIVED.sha256"


def derived_paths():
    out = [plans.PLANS / "S8.json"]
    for sid in plans.BRIEFS:
        out += [plans.PLANS / f"{sid}.{kind}.json" for kind in ("occurrences", "asks", "final-occurrences", "ledger")]
    return out


def regenerate():
    plans.write_twins()
    for sid in plans.BRIEFS:
        plan = json.loads((plans.PLANS / f"{sid}.json").read_text())
        if plans.validate(plan):
            raise ValueError(f"{sid}: plan does not validate")
        occ = keys.occurrences(plan)
        (plans.PLANS / f"{sid}.occurrences.json").write_text(json.dumps(occ, indent=1, ensure_ascii=False) + "\n")
        phrasing = json.loads((plans.PLANS / f"{sid}.questions.json").read_text())
        problems = phrase.check(plan, occ, phrasing)
        if problems:
            raise ValueError(f"{sid}: questions do not check: {problems[:3]}")
        (plans.PLANS / f"{sid}.asks.json").write_text(json.dumps(phrase.asks(plan, occ, phrasing), indent=1, ensure_ascii=False) + "\n")
        (plans.PLANS / f"{sid}.final-occurrences.json").write_text(
            json.dumps(phrase.final_occurrences(plan, occ, phrasing), indent=1, ensure_ascii=False) + "\n")
        (plans.PLANS / f"{sid}.ledger.json").write_text(json.dumps(phrase.plan_ledger(plan), indent=1, ensure_ascii=False) + "\n")


def digests():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in derived_paths()}


def main() -> int:
    regenerate()
    now = digests()
    if "--pin" in sys.argv or not PINS.exists():
        PINS.write_text("".join(f"{h}  {n}\n" for n, h in sorted(now.items())))
        print(f"pinned {len(now)} derived files")
        return 0
    pinned = dict(line.split("  ")[::-1] for line in PINS.read_text().split("\n") if line.strip())
    differ = sorted(n for n, h in now.items() if pinned.get(n) != h)
    if differ:
        print(f"derived files differ from their pins: {differ}", file=sys.stderr)
        return 1
    print(f"{len(now)} derived files regenerated, identical to their pins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
