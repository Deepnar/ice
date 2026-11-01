"""FINAL's no-memory controls where no question chain writes them (pre-registration revision 5): the
LoCoMo conversations and the synthetic S1–S8. Z2's control machine (questions.controls_at): four
kinds cycled — general knowledge, a self-contained task, small talk, a follow-up answerable from the
recent turns — written in the conversation's style by one writer family and kept only when the
other confirms it needs no long-term memory. The private histories get theirs from their own chain.

- LoCoMo: PER controls after the last turn (where its questions are asked), each writer shown the
  ones already kept; written to logs/final-locomo/<conv>/controls.json and merged into its
  asks.json (the export merges them too), so ICE's read and the memory systems' search take them.
- Synthetic: one per checkpoint of the plan, once the conversation is generated; written to
  logs/final-synth/<id>/controls.json, and logs/final-synth/<id>/asks.json = the plan's asks + them.

Cloud calls (DeepSeek writes, MiMo checks, and the reverse), durable and resumable.

    uv run --no-sync python experiments/final/no_memory_controls.py locomo
    uv run --no-sync python experiments/final/no_memory_controls.py synth --id S1
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import questions as Q  # noqa: E402

LOCOMO, SYNTH = ROOT / "logs/final-locomo", ROOT / "logs/final-synth"
PER = 10                               # per LoCoMo conversation: ~100 in all (revision 5)
SYNTHETIC = tuple(f"S{i}" for i in range(1, 9))


def models():
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    return {r: (generator(r).generate, metadata(r)) for r in Q.WRITERS_CYCLE}


def merged(asks, kept):
    """The asks with the controls replaced by *kept* (rerunning never duplicates them)."""
    return [a for a in asks if a.get("kind", "question") != "control"] + Q.control_asks(kept)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def locomo(base=LOCOMO, *, per=PER, models_=None):
    done = {}
    for d in sorted(p for p in Path(base).iterdir() if (p / "originals.json").exists()):
        originals = json.loads((d / "originals.json").read_text())
        last = originals[-1]["turn_number"]
        kept, rejected = Q.controls_at(f"LOCOMO-{d.name}", originals, d / "controls", [last] * per,
                                       models=models_ or models(), avoid=True)
        write(d / "controls.json", dict(policy=Q.CONTROL_POLICY, kept=kept, rejected=rejected))
        write(d / "asks.json", merged(json.loads((d / "asks.json").read_text()), kept))
        done[d.name] = dict(kept=len(kept), rejected=len(rejected))
    return done


def synth(sid, base=SYNTH, *, models_=None):
    from experiments.synth.keys import checkpoints
    from experiments.synth.plans import PLANS
    d = Path(base) / sid
    plan = json.loads((PLANS / f"{sid}.json").read_text())
    originals = json.loads((d / "originals.json").read_text())
    kept, rejected = Q.controls_at(sid, originals, d / "controls", checkpoints(plan), models=models_ or models())
    write(d / "controls.json", dict(policy=Q.CONTROL_POLICY, kept=kept, rejected=rejected))
    write(d / "asks.json", merged(json.loads((PLANS / f"{sid}.asks.json").read_text()), kept))
    return dict(kept=len(kept), rejected=len(rejected))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("locomo")
    s = sub.add_parser("synth")
    s.add_argument("--id", required=True, choices=SYNTHETIC)
    args = parser.parse_args()
    print(json.dumps(locomo() if args.command == "locomo" else synth(args.id), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
