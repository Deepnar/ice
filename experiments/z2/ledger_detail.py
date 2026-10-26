"""Z2 S1b: a per-reply detail pass over long assistant replies.

The window extraction (25 pairs per call) records the gist of long assistant replies
and drops their specifics: on the development history the merged ledger covered
48-50 of 81 reviewed requirements, the misses mostly items inside long advice. Here
each long reply is read on its own: one writer family (alternating by turn) is shown
the reply and what the ledger already records from it, and lists every specific the
ledger lacks; the OTHER family checks each new entry with the merge's blind
observation (said_by / stated_as / wrong_detail, compared in code); disputes go to
the tie-breaker. Entries keep the merge's statuses; nothing is deleted.

    uv run --no-sync python experiments/z2/ledger_detail.py --history dev-ecc64aab \
        --ledger logs/z2-ledger-dev-<date>-r4/dev-ecc64aab --out logs/z2-ledger-detail-<date>-r1
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import ledger as L
from experiments.z2 import ledger_merge as M

POLICY = "ice-z2-ledger-detail-1"
LONG_REPLY_WORDS = 500
DETAIL_SYSTEM = """You add detail to a fact ledger built from a conversation between a user and an AI
assistant. TURN is one long assistant reply. EXISTING lists what the ledger already records from it.
List every specific item in the reply that EXISTING does not already carry: each recommendation,
option, step, reason, direction, criterion, warning, number, name or decision — ONE entry per item,
with its specifics and any condition attached ("because…", "only if…").
Each entry:
- kind: one of fact, preference, decision, plan, advice, procedure, event
- subject: a short name for what it is about
- value: one self-contained sentence stating the item fully
- speaker: "assistant" (or "user" if the item is a user statement quoted in the reply's context)
- modality: asserted, suggested (proposed, not adopted), or hypothetical
- citations: 1 to 3 objects {turn, role, quote}: quote is an exact contiguous excerpt of 3 to 40
  words copied character for character from the reply; never fix typos or join pieces
- supersedes: null; change: null
Return only JSON: {"entries": [ ... ]}; {"entries": []} if EXISTING already carries everything."""


def halves(text):
    """Two halves of a reply, split at the paragraph break nearest the middle (else a
    sentence end, else the middle). Both stay exact substrings of the original."""
    import re
    middle = len(text) // 2
    for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+"):
        cuts = [m.end() for m in re.finditer(pattern, text)]
        if cuts:
            cut = min(cuts, key=lambda c: abs(c - middle))
            if 0 < cut < len(text):
                return text[:cut], text[cut:]
    return text[:middle], text[middle:]


MAX_SPLITS = 3


def long_turns(originals, words=LONG_REPLY_WORDS):
    return [r["turn_number"] for r in originals if len(r["response"].split()) > words]


def run(history, ledger_dir, out, *, models, writers=("deepseek", "mimo"), tie_breaker="muse",
        limit=None):
    originals = L.load_history(history)
    ledger = json.loads((ledger_dir / "ledger.json").read_text())
    entries = ledger["entries"]
    out.mkdir(parents=True, exist_ok=True)
    calls = M.Calls(out / "detail-raw.jsonl")
    session = lambda role: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{history}|{role}"))
    added, queue = [], []
    stats = dict(turns=0, proposed=0, admitted=0, dropped=0, verified=0, tie_kept=0, unconfirmed=0,
                 split_replies=0, unfit_parts=0, failed_turns=[])
    targets = long_turns(originals)[:limit] if limit else long_turns(originals)
    for n, turn in enumerate(targets):
        stats["turns"] += 1
        writer = writers[n % 2]
        checker = writers[(n + 1) % 2]
        existing = [e["value"] for e in entries if turn in {c["turn"] for c in e["citations"]}]

        def propose(text, depth=0):
            # A reply listing more items than the output cap allows is asked again in
            # halves of the reply (2026-10-19: one B reply was cut and stopped the run).
            prompt = json.dumps(dict(TURN=dict(turn=turn, role="assistant", text=text), EXISTING=existing),
                                ensure_ascii=False)
            try:
                got, _ = M.ask(calls, writer, models[writer][1], models[writer][0], DETAIL_SYSTEM, prompt,
                               "entries", session(writer))
                return got or []
            except Exception as exc:                          # noqa: BLE001
                if not M._cut(exc):
                    raise
                if depth >= MAX_SPLITS:
                    stats["unfit_parts"] += 1
                    return []
                stats["split_replies"] += depth == 0
                first, second = halves(text)
                return propose(first, depth + 1) + propose(second, depth + 1)

        try:
            rows = propose(L.role_text(originals, turn, "assistant"))
        except Exception as exc:                              # noqa: BLE001 — this turn only
            stats["failed_turns"].append(dict(turn=turn, error=f"{type(exc).__name__}: {exc}"[:200]))
            continue
        window = dict(new=[turn], context=[])
        fresh = []
        for k, raw in enumerate(rows or []):
            stats["proposed"] += 1
            entry, reason = L.admit(originals, raw, window, set(), f"{writer}:{history}:d{turn}.{k}")
            if entry is None:
                stats["dropped"] += 1
                continue
            stats["admitted"] += 1
            fresh.append(dict(entry, history=history, writer=writer, members=[entry["id"]],
                              supersedes_raw=[], detail_pass=True))
        if not fresh:
            continue
        record = dict(context=[], new=[turn])
        check = M.aliases(fresh, "C")
        answered, _ = M.ask_covering(calls, checker, models[checker][1], models[checker][0],
                                     M.VERIFY_SYSTEM, lambda sub: M.verify_prompt(originals, record, sub, {}),
                                     "checks", session(checker), check)
        disputed = []
        for alias, e in check.items():
            supported, reason = M.observed(e, answered[alias]) if alias in answered else (None, "missing")
            if supported:
                added.append(dict(e, status="verified", verified=True, usable_for_questions=True))
                stats["verified"] += 1
            else:
                disputed.append((e, reason))
        if disputed and tie_breaker is None:            # FINAL (Z3 D13): disputes are not settled by a third model
            for e, reason in disputed:
                added.append(dict(e, status="unconfirmed", verified=False, usable_for_questions=False,
                                  unconfirmed_reason="disputed"))
                stats["unconfirmed"] += 1
        elif disputed:
            asked = M.aliases([e for e, _ in disputed], "D")
            decided, _ = M.ask_covering(calls, tie_breaker, models[tie_breaker][1], models[tie_breaker][0],
                                        M.TIE_SYSTEM, lambda sub: M.tie_prompt(originals, record, sub),
                                        "decisions", session(tie_breaker), asked)
            for alias, e in asked.items():
                reason = next(r for d, r in disputed if d is e)
                keep = M.observed(e, decided[alias])[0] if alias in decided else None
                if keep:
                    added.append(dict(e, status="tie_break", verified=False, usable_for_questions=True))
                    stats["tie_kept"] += 1
                else:
                    added.append(dict(e, status="unconfirmed", verified=False, usable_for_questions=False,
                                      unconfirmed_reason=reason or "tie_break_unanswered"))
                    stats["unconfirmed"] += 1
                queue.append(dict(kind="detail_tie_break", entry=e, checker_reason=reason, kept=bool(keep)))
    combined = dict(ledger, entries=entries + added, detail_pass=dict(policy=POLICY, stats=stats))
    (out / "ledger.json").write_text(json.dumps(combined, ensure_ascii=False, indent=1) + "\n")
    (out / "detail-queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(stats))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--ledger", type=Path, required=True, help="directory holding ledger.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tie-breaker", default="muse", help="a role, or none (FINAL: disputes unconfirmed)")
    args = parser.parse_args()
    tie_breaker = None if args.tie_breaker == "none" else args.tie_breaker
    for path in (args.ledger, args.out):
        if not path.resolve().is_relative_to(ROOT / "logs"):
            raise ValueError("ledger data is private: keep it under logs/")
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    models = {r: (generator(r).generate, metadata(r)) for r in ("deepseek", "mimo") + ((tie_breaker,) if tie_breaker else ())}
    run(args.history, args.ledger, args.out, models=models, limit=args.limit, tie_breaker=tie_breaker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
