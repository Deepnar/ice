"""Fact-replacement designs through the production path (Z2 F20; RETESTS R8).

For each codex_fact_reconcile_design (pre_g60, g60, context): Z2 S8's 99 judged replacements
through facts.reconcile, and the CODEX_V4 qualification set through facts.screen + facts.reconcile
(production's batch screen then single-pair question), with fact_extractor's own model calls.
The "context" design gets what production passes: each fact's cited quote and its date (S8: the
best-matching window of the audit's source turn, dated on the replay clock; qualification set:
the item's quote and timestamp).

Acceptance (the G60 bar): invalid kept <= 10/53, valid kept >= 38/46, real >= 89/99, planted
>= 10/12, distractors 0/60. The S8 labels are partly inconsistent (FINDINGS F20), so the bar
may be out of reach; the design with the fewest errors on both sets is the candidate.

    uv run --no-sync python experiments/z2/reconcile_eval.py
"""

import collections
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "logs/z2-s0-reconcile-batch-2026-10-18-r1"))

from experiments.z2.supersession_designs import window  # noqa: E402


def s8_set():
    from experiments.z2.grade import turn_dates
    s8 = json.loads((ROOT / "logs/fix-pass/supersession-s8-pairs.json").read_text())
    dates = {h: turn_dates(h) for h in ("B", "C")}
    src = {}
    for h in ("B", "C"):
        for i in json.loads((ROOT / f"logs/z2-run-main/s8-{h}/items.json").read_text())["items"]:
            if i["kind"] == "fact_supersession":
                s = i["sources"] if isinstance(i["sources"], list) else json.loads(i["sources"])
                s = sorted(s, key=lambda x: x["turn"])
                if s:
                    src[i["id"]] = (h, s[0], s[-1])
    out = []
    for r in s8:
        p = None
        if r["id"] in src:
            h, a, b = src[r["id"]]
            p = dict(earlier=dict(said=dates[h].get(a["turn"]), passage=window(a, r["earlier"])["passage"]),
                     newer=dict(said=dates[h].get(b["turn"]), passage=window(b, r["newer"])["passage"]))
        out.append((r, p))
    return out


def run(design):
    from src.api.config import settings
    from src.memory import facts
    from src.workers import fact_extractor as X
    settings.codex_fact_reconcile_design = design

    def single(earlier, newer, passages=None, when=None):
        return facts.reconcile({"text": newer, "modality": "asserted"},
                               NS(text=earlier, modality="asserted", mentioned_at=None),
                               ask=X._ask, when=when, passages=passages)

    s8 = s8_set()
    got = [single(r["earlier"], r["newer"], p) for r, p in s8]
    kept = lambda valid: sum(1 for (r, _), g in zip(s8, got) if r["valid"] == valid and g == "replaces")
    line = f"{design:8} S8 valid kept {kept(True)}/46, invalid kept {kept(False)}/53"

    import qualify_batch as B
    it = json.loads((ROOT / B.L / "dedupe-candidates.json").read_text())["items"]
    pairs = json.loads((ROOT / B.L / "dedupe-label-pairs.json").read_text())
    cited = lambda x: dict(said=(x.get("ts") or "")[:16] or None, passage=x.get("quote") or x["text"])
    bare = lambda text: dict(said=None, passage=text)

    def hybrid(newer, earlier, newer_ctx):
        needs = facts.screen({"text": newer}, [NS(text=e["text"]) for e in earlier], ask_many=X._ask_many)
        return [single(e["text"], newer, dict(earlier=e["ctx"], newer=newer_ctx)) if need else "both_true"
                for e, need in zip(earlier, needs)]

    groups = collections.defaultdict(list)
    for n, p in enumerate(pairs):
        if n not in B.G.EXCLUDE:
            groups[p["new"]].append((n, p["old"]))
    ok = lambda got, gold: (got == gold or (gold == "not_same" and got != "same")
                            or (gold == "not_replaces" and got != "replaces"))
    right = total = 0
    misses = collections.Counter()                    # (gold, got) of the real pairs it got wrong
    for new, members in groups.items():
        v = hybrid(it[new]["text"], [dict(text=it[o]["text"], ctx=cited(it[o])) for _, o in members], cited(it[new]))
        for (n, _), x in zip(members, v):
            gold = B.G.REAL.get(n, "both_true")
            right += ok(x, gold); total += 1
            if not ok(x, gold):
                misses[f"{gold}->{x}"] += 1
    rng = random.Random(20261018)
    pool = sorted({p["old"] for p in pairs})
    planted_right = distract = 0
    for old, new, gold, _note in B.G.PLANTED:
        others = [dict(text=it[o]["text"], ctx=cited(it[o])) for o in rng.sample(pool, 5)]
        pos = rng.randrange(6)
        v = hybrid(new, others[:pos] + [dict(text=old, ctx=bare(old))] + others[pos:], bare(new))
        planted_right += ok(v[pos], gold)
        distract += sum(x == "replaces" for i, x in enumerate(v) if i != pos)
    line += (f" | real {right}/{total}, planted {planted_right}/{len(B.G.PLANTED)}, distractors replaced {distract}/60"
             f" | real misses {dict(misses)}")
    return line


def main() -> int:
    for design in ("pre_g60", "g60", "context"):
        print(run(design), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
