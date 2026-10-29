"""The "context" fact-replacement design's evidence variants, through the production path
(Z2 F20; RETESTS R8). pre_g60 and g60 are already measured (FINDINGS F20) and not re-run.

For each codex_fact_reconcile_context (quote, window, turn, fields):
- Z2 S8's 99 judged replacements, rebuilt from B's and C's stores: the earlier fact, the fact
  that replaced it, each one's first source (quote, date, source turn) — inputs made by
  facts.passages, exactly as write_turn_facts makes them — then facts.reconcile;
- the CODEX_V4 qualification set (99 real pairs, 12 planted with 5 distractors each) through
  facts.screen + facts.reconcile, evidence from facts.context_passage with each item's quote,
  fields, timestamp and its original turn in the development history.

Acceptance (the G60 bar): invalid kept <= 10/53, valid kept >= 38/46, real >= 89/99, planted
>= 10/12, distractors 0/60; the S8 labels are partly inconsistent, so the variant with the
fewest errors on both sets is the candidate. Each line also gives the mean evidence length.

    uv run --no-sync python experiments/z2/reconcile_eval.py                 # all four variants
    uv run --no-sync python experiments/z2/reconcile_eval.py --modes window --limit 3   # a smoke check
"""

import argparse
import collections
import json
import random
import statistics as S
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "logs/z2-s0-reconcile-batch-2026-10-18-r1"))

STORES = {"B": "ice_z2_main_b", "C": "ice_z2_main_c"}


def chars(p):
    return sum(len(str(v)) for side in p.values() for k, v in side.items() if k != "said") if p else 0


def s8_cases(limit=None):
    """[(valid, earlier row, claim dict, assertion dict, newer turn row, when, session)] from the stores."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.api.config import settings
    from src.memory.models import CodexFact, CodexFactSource, EpisodicMemory
    s8 = json.loads((ROOT / "logs/fix-pass/supersession-s8-pairs.json").read_text())[:limit]
    sessions = {h: Session(create_engine(settings.database_url.rsplit("/", 1)[0] + "/" + db)) for h, db in STORES.items()}
    out = []
    for r in s8:
        db = sessions[r["history"]]
        earlier = db.get(CodexFact, r["id"].split(":", 1)[1])
        newer = db.get(CodexFact, earlier.superseded_by) if earlier is not None and earlier.superseded_by else None
        if newer is None:
            print(f"  missing in store: {r['id']}", flush=True)
            continue
        src = (db.query(CodexFactSource).filter(CodexFactSource.fact_id == newer.id)
               .order_by(CodexFactSource.recorded_at).first())
        claim = {k: getattr(newer, k) for k in ("text", "modality", "speaker", "subject", "relation", "object")}
        turn = db.get(EpisodicMemory, src.episodic_id) if src is not None and src.episodic_id else None
        out.append((r["valid"], earlier, claim, {"quote": getattr(src, "quote", "")},
                    turn, getattr(src, "recorded_at", None) or newer.mentioned_at, db))
    return out


def qualification():
    import qualify_batch as B
    from experiments.z2.ledger import load_history, role_text
    it = json.loads((ROOT / B.L / "dedupe-candidates.json").read_text())["items"]
    pairs = json.loads((ROOT / B.L / "dedupe-label-pairs.json").read_text())
    dev = load_history("dev-ecc64aab")
    turn = lambda n: (f"User: {role_text(dev, n, 'user')}\n\nAssistant: {role_text(dev, n, 'assistant')}",
                      role_text(dev, n, "user")) if n and 1 <= n <= len(dev) else ("", "")
    return B, it, pairs, turn


def run(mode, limit=None):
    from src.api.config import settings
    from src.memory import facts
    from src.workers import fact_extractor as X
    settings.codex_fact_reconcile_design, settings.codex_fact_reconcile_context = "context", mode
    lengths = []

    def decide(claim, earlier, passages, when=None):
        lengths.append(chars(passages))
        return facts.reconcile(claim, earlier, ask=X._ask, when=when, passages=passages)

    cases = s8_cases(limit)
    got = []
    for valid, earlier, claim, assertion, turn, when, db in cases:
        p = facts.passages(db, earlier, assertion, claim, when, turn)
        got.append((valid, decide(claim, earlier, p, when)))
    kept = lambda v: sum(1 for valid, g in got if valid == v and g == "replaces")
    n_valid = sum(1 for v, _ in got if v)
    line = f"{mode:7} S8 valid kept {kept(True)}/{n_valid}, invalid kept {kept(False)}/{len(got) - n_valid}"

    B, it, pairs, turn_of = qualification()

    def evidence(x):
        t, u = turn_of(int(x["turn"]) if str(x.get("turn", "")).isdigit() else None)
        return facts.context_passage(mode, quote=x.get("quote") or x["text"], said=facts.utc_stamp(x.get("ts")),
                                     turn_text=t, user_text=u, fields=facts._fields(x))
    bare = lambda text: facts.context_passage(mode, quote=text, said=None)
    fact = lambda x: NS(text=x["text"], modality=x.get("modality", "asserted"), mentioned_at=None)

    def hybrid(newer, newer_ev, earlier):
        needs = facts.screen({"text": newer["text"]}, [NS(text=e["text"]) for e, _ in earlier], ask_many=X._ask_many)
        return [decide(dict(newer, text=newer["text"]), fact(e), dict(earlier=ev, newer=newer_ev)) if need else "both_true"
                for (e, ev), need in zip(earlier, needs)]

    groups = collections.defaultdict(list)
    for n, p in enumerate(pairs):
        if n not in B.G.EXCLUDE:
            groups[p["new"]].append((n, p["old"]))
    ok = lambda got, gold: (got == gold or (gold == "not_same" and got != "same")
                            or (gold == "not_replaces" and got != "replaces"))
    right = total = 0
    misses = collections.Counter()
    for new, members in list(groups.items())[:limit]:
        v = hybrid(it[new], evidence(it[new]), [(it[o], evidence(it[o])) for _, o in members])
        for (n, _), x in zip(members, v):
            gold = B.G.REAL.get(n, "both_true")
            right += ok(x, gold); total += 1
            if not ok(x, gold):
                misses[f"{gold}->{x}"] += 1
    rng = random.Random(20261018)
    pool = sorted({p["old"] for p in pairs})
    planted_right = distract = 0
    planted = B.G.PLANTED[:limit] if limit else B.G.PLANTED
    for old, new, gold, _note in planted:
        others = [(it[o], evidence(it[o])) for o in rng.sample(pool, 5)]
        pos = rng.randrange(6)
        v = hybrid({"text": new, "modality": "asserted"}, bare(new),
                   others[:pos] + [({"text": old, "modality": "asserted"}, bare(old))] + others[pos:])
        planted_right += ok(v[pos], gold)
        distract += sum(x == "replaces" for i, x in enumerate(v) if i != pos)
    return (line + f" | real {right}/{total}, planted {planted_right}/{len(planted)}, distractors replaced "
            f"{distract}/{5 * len(planted)} | evidence chars per call ~{int(S.mean(lengths)) if lengths else 0}"
            f" | real misses {dict(misses)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modes", default="quote,window,turn,fields")
    parser.add_argument("--limit", type=int, help="first N of each set (a smoke check)")
    args = parser.parse_args()
    for mode in args.modes.split(","):
        print(run(mode, args.limit), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
