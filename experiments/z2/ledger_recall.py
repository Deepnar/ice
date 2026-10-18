"""Z2 validation item 1 (spec §5): how many reviewed development facts the ledger finds.

For each of the 67 reviewed development questions (all on dev history ecc64aab), the
ledger entries citing any of its gold turns are shown to the judge (Muse — never a
writer) with each reviewed requirement; the judge reports whether the entries state
it (meaning, not wording). Recall = requirements covered / requirements, reported
by the best status among covering entries (verified / tie_break / relabeled /
unconfirmed). Each call is durable and resumable.

    uv run --no-sync python experiments/z2/ledger_recall.py \
        --ledger logs/z2-ledger-dev-<date>-r2/dev-ecc64aab --out logs/z2-ledger-recall-<date>-r1
"""

import argparse
import collections
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import ledger_merge as M
from experiments.z2.questions import ask_object

POLICY = "ice-z2-ledger-recall-1"
CONTRACTS = ROOT / "logs/v3-content-contract-ledger-2026-10-13-r1/ledger.json"
KEYS = ROOT / "logs/v3-offline-input-plan-2026-10-10-r4/development-keys.json"
STATUS_ORDER = ("verified", "relabeled", "tie_break", "unconfirmed")
SYSTEM = """A reviewer wrote a REQUIREMENT: information a correct answer must contain about a
conversation. ENTRIES are facts extracted from that conversation. Report whether the entries
together state the information the requirement asks for (meaning, not wording), and which
entry ids carry it.
Return only JSON: {"covered": true|false, "entry_ids": ["..."], "reason": "..."}"""


def contracts():
    keys = {r["review"]["probe_id"]: r for r in json.loads(KEYS.read_text())["records"]}
    out = []
    for c in json.loads(CONTRACTS.read_text())["records"]:
        k = keys.get(c["probe_id"])
        if k and k["review"]["conversation"] == "ecc64aab":
            out.append(dict(probe_id=c["probe_id"], gold_turns=k["gold_turns"],
                            requirements=c["requirements"], question=k["review"]["question"]))
    return out


def run(ledger_dir, out, *, judge="muse", generate=None, meta=None):
    entries = json.loads((ledger_dir / "ledger.json").read_text())["entries"]
    out.mkdir(parents=True, exist_ok=True)
    calls = M.Calls(out / "recall-raw.jsonl")
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{judge}"))
    rows, tally = [], collections.Counter()
    for c in contracts():
        gold = set(c["gold_turns"])
        cited = [e for e in entries if gold & {x["turn"] for x in e["citations"]}]
        shown = {f"E{n}": e for n, e in enumerate(cited, 1)}
        for req in c["requirements"]:
            if not shown:
                verdict, status = {"covered": False, "reason": "no entry cites a gold turn"}, None
            else:
                verdict = ask_object(calls, judge, meta, generate, SYSTEM, json.dumps(dict(
                    REQUIREMENT=req, ENTRIES=[dict(id=a, speaker=e["speaker"], text=e["value"])
                                              for a, e in shown.items()]), ensure_ascii=False),
                    session, max_output_tokens=12_000) or {}       # Muse reasons before answering
                ids = [shown[a] for a in verdict.get("entry_ids") or [] if a in shown]
                status = min((e["status"] for e in ids), key=STATUS_ORDER.index, default=None)
            covered = verdict.get("covered") is True
            tally["requirements"] += 1
            tally["covered"] += covered
            if covered:
                tally[f"covered_by_{status}"] += 1
            if not shown:
                tally["no_entry_on_gold_turns"] += 1
            rows.append(dict(probe_id=c["probe_id"], requirement=req, entries_on_gold=len(shown),
                             covered=covered, best_status=status, reason=verdict.get("reason")))
    result = dict(policy=POLICY, judge=judge, judge_meta=meta, tally=dict(tally), rows=rows)
    (out / "recall.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(tally)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.ledger, args.out):
        if not path.resolve().is_relative_to(ROOT / "logs"):
            raise ValueError("private data: keep it under logs/")
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    run(args.ledger, args.out, generate=generator("muse").generate, meta=metadata("muse"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
