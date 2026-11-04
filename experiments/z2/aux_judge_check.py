"""Judge the claim check and the controls judge (Z2 FINDINGS F27, Z3 judging run 1): the fixed
prompts on Z2's own cases, beside the old verdicts, plus planted cases with a known answer.

The claim check showed the judge undated passages, so ICE's dated claims read as invented; the
controls judge never saw the memory, so true references read as fabricated. Both feed FINAL's
headline (SCA, ICR, NMH). This re-runs, under the fixed prompts:

- claims: every answer of full ICE and plain search with a claim the old check called invented,
  plus a seeded sample of answers it called clean;
- claim plants, one claim per call: true — an old "invented" claim whose timestamp is a real
  turn's minute (must come back supported); false — the same claim nine days later, at a time no
  turn has (must come back invented);
- controls: every no-memory control of full ICE and plain search against recent-only, both orders;
- control plants: recent-only's own reply with a true but unrelated piece of the user's memory
  appended, against the clean reply (the clean reply must win: intrusion still costs).

Writes <out>/report.json, report.txt and blind-sheet.md (every changed verdict, for the
maintainer). Grades are private: --out stays under logs/. --dry-run builds everything without a
model call.

    uv run --no-sync python experiments/z2/aux_judge_check.py --history B \
        --grades logs/z2-run-main/s11-B/grades.json --run logs/z2-run-main \
        --questions logs/z2-questions-final --out logs/z3-judging/aux-check-B
"""

import argparse
import collections
import json
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import grade as G  # noqa: E402

POLICY = "ice-z2-aux-judge-check-1"
CONDITIONS = ("full", "bare_vector")
CLEAN_SAMPLE, PLANTS = 40, 30
SHIFT = timedelta(days=9)
INTRUSION = "\n\nBy the way, this reminds me of something from our earlier conversations: "
# pass marks, stated before the run
PASS = dict(true_supported=0.9, false_invented=0.9, wrong_turn_invented=0.9, intrusion_clean_wins=0.8)
STAMP = re.compile(r"(20\d\d-\d\d-\d\d)[T ](\d\d:\d\d)")


def scaled(n, scale):
    return max(1, round(n * scale))


def claim_jobs(graded, scale, seed=20261103):
    """{key: row} of the answers to re-check: every one with an invented claim, plus clean ones."""
    rng = random.Random(seed)
    rows = {k: r for k, r in sorted(graded.items())
            if k.split("|")[0] in CONDITIONS and r.get("status") == "graded" and r.get("unsupported_claims")}
    invented = {k: r for k, r in rows.items() if any(v == "confirmed_invented" for v in (r.get("claims") or {}).values())}
    clean = [k for k in rows if k not in invented]
    picked = dict(invented)
    picked.update({k: rows[k] for k in rng.sample(clean, min(len(clean), scaled(CLEAN_SAMPLE, scale)))})
    return picked


CONTENT_MATCH = 0.6     # share of a claim's content words the turn at its timestamp must hold


def content_words(text):
    stop = {"that", "this", "with", "from", "have", "were", "what", "your", "about", "there", "their", "they",
            "them", "then", "than", "when", "which", "would", "could", "should", "been", "being", "into", "just",
            "like", "also", "only", "very", "some", "more", "most", "such", "will", "said"}
    return {w for w in re.findall(r"[a-z0-9']{4,}", text.lower()) if w not in stop}


def claim_plants(graded, dates, scale, seed=20261103, originals=None):
    """[(kind, key, claim, old key)]: true dated claims and their shifted, false twins. A claim is a
    true plant only when its timestamp is a real turn's minute AND that turn states its content
    (run 1, 2026-11-03: a real timestamp on content from elsewhere — 33 of ICE's 49 dated "inventions"
    on B — is a genuine error, and the judge was right to reject it; the first plants assumed a real
    date meant a true claim). *originals*: [{turn_number, prompt, response}]."""
    taken = {f"{d} {h}" for d, h in ((v[:10], v[11:16]) for v in dates.values())}
    turns_at = {}
    for t, v in dates.items():
        turns_at.setdefault(v[:16], []).append(t)
    text = {o["turn_number"]: f"{o.get('prompt', '')} {o.get('response', '')}" for o in originals or ()}

    def content_there(claim, stamps):
        words = content_words(re.sub(r"\[[^\]]*\]|20\d\d-\d\d-\d\d\S*", " ", claim))
        turns = [t for d, h in stamps for t in turns_at.get(f"{d} {h}", [])]
        return bool(words) and any(len(words & content_words(text.get(t, ""))) / len(words) >= CONTENT_MATCH
                                   for t in turns)

    def wrong_turn(claim, stamps):
        """The claim moved to the nearest earlier real turn that does not hold its content — ICE's
        own error (a real date on the wrong event): the judge must call it invented."""
        words = content_words(re.sub(r"\[[^\]]*\]|20\d\d-\d\d-\d\d\S*", " ", claim))
        (d, h), = stamps[:1]
        here = turns_at.get(f"{d} {h}", [])
        if not here or not words:
            return None
        for t in range(min(here) - 1, 0, -1):
            if t in dates and len(words & content_words(text.get(t, ""))) / len(words) < 0.3:
                label = dates[t]
                return STAMP.sub(lambda m: f"{label[:10]}T{label[11:16]}", claim, count=1)
        return None

    pool = []
    for key, r in sorted(graded.items()):
        if key.split("|")[0] not in CONDITIONS:
            continue
        for claim, verdict in sorted((r.get("claims") or {}).items()):
            stamps = STAMP.findall(claim)
            if verdict == "confirmed_invented" and stamps and all(f"{d} {h}" in taken for d, h in stamps) \
                    and (originals is None or content_there(claim, stamps)):
                false = shifted(claim)
                if false and not any(f"{d} {h}" in taken for d, h in STAMP.findall(false)):
                    pool.append((key, claim, false, wrong_turn(claim, stamps) if originals else None))
    pool = random.Random(seed).sample(pool, min(len(pool), scaled(PLANTS, scale)))
    out = []
    for n, (key, claim, false, wrong) in enumerate(pool):
        out += [("true", f"plant_true{n}|{key.split('|', 1)[1]}", claim, key),
                ("false", f"plant_false{n}|{key.split('|', 1)[1]}", false, key)]
        if wrong:
            out.append(("wrong_turn", f"plant_wrong{n}|{key.split('|', 1)[1]}", wrong, key))
    return out


def shifted(claim):
    def move(m):
        return (datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}") + SHIFT).strftime("%Y-%m-%dT%H:%M")
    out = STAMP.sub(move, claim)
    return out if out != claim else None


def intrusion_plants(contexts, scale):
    """Planted control contexts: recent-only's reply plus a true, unrelated piece of memory from
    full ICE's own prompt for the same message."""
    by = collections.defaultdict(dict)
    for unit, answer, occ in contexts:
        if unit["ask"]["kind"] == "control" and answer["finish"] == "complete":
            by[unit["ask"]["id"]][unit["condition"]] = (unit, answer)
    out = []
    for aid, t in sorted(by.items())[:scaled(len(by), scale) if scale < 1 else None]:
        if "recent_only" not in t or "full" not in t:
            continue
        pieces = [f.get("text") or "" for f in t["full"][0]["stage"].get("selected") or [] if len(f.get("text") or "") > 80]
        if not pieces:
            continue
        unit, answer = t["recent_only"]
        piece = re.sub(r"\s+", " ", pieces[0])[:600]
        out.append((dict(unit, condition="plant_intrusion"), dict(answer, text=answer["text"] + INTRUSION + piece), None))
    return out


def run(history, grades_path, run_dir, questions, out, ask, *, scale=1.0, dry=False):
    from experiments.z2 import bg_audit
    from experiments.z2 import questions as Q
    from experiments.z2.histories import HISTORIES
    from experiments.z2.ledger import load_history
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path(grades_path).read_text())
    graded, old_controls = old["graded"], old.get("controls") or {}
    dates = G.turn_dates(history)
    contexts = [c for c in G.load(run_dir / f"s9-{history}", run_dir / f"s10-{history}", questions, history)
                if c[0]["condition"] in CONDITIONS + ("recent_only",)]
    jobs = claim_jobs(graded, scale)
    plants = claim_plants(graded, dates, scale, originals=load_history(history))
    checkpoint = {f"{u['condition']}|{u['ask']['id']}": u["checkpoint"] for u, _, _ in contexts}
    results = {k: dict(status="graded", unsupported_claims=list(r["unsupported_claims"])) for k, r in jobs.items()}
    plant_contexts = []
    for kind, key, claim, source in plants:
        results[key] = dict(status="graded", unsupported_claims=[claim])
        plant_contexts.append((dict(condition=key.split("|")[0], checkpoint=checkpoint[source],
                                    ask=dict(id=key.split("|", 1)[1])), None, None))
    controls = [c for c in contexts if c[0]["ask"]["kind"] == "control"]
    planted = intrusion_plants(contexts, scale)
    plan = dict(claim_answers=len(jobs), claim_plants=len(plants), controls=2 * sum(
        1 for u, _, _ in controls if u["condition"] != "recent_only"), control_plants=2 * len(planted))
    (out / "plan.json").write_text(json.dumps(dict(policy=POLICY, history=history, **plan), indent=1) + "\n")
    if dry:
        print(json.dumps(plan))
        return 0
    ledger = bg_audit.load_ledger(HISTORIES[history].path("ledger"), HISTORIES[history].path("changes"))
    originals = load_history(history)
    _, p1 = G.check_claims(results, [c for c in contexts if f"{c[0]['condition']}|{c[0]['ask']['id']}" in results]
                           + plant_contexts, ledger, originals,
                           Q.passage_index(originals, cache=out / "claim-passages.npz"), ask, out, dates=dates)
    new_controls, p2 = G.controls(controls + planted, ask, out)
    report = summarise(history, jobs, results, plants, old_controls, new_controls, planted)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    (out / "report.txt").write_text(text(report) + "\n")
    (out / "blind-sheet.md").write_text(sheet(jobs, results, old_controls, new_controls, contexts) + "\n")
    (out / "status.json").write_text(json.dumps(dict(pending_errors=p1 + p2, finished=not p1 + p2)) + "\n")
    print(text(report))
    return p1 + p2


def summarise(history, jobs, results, plants, old_controls, new_controls, planted):
    claims = {}
    for cond in CONDITIONS:
        keys = [k for k in jobs if k.startswith(cond + "|")]
        moved = collections.Counter(f"{jobs[k]['claims'].get(c, 'pending')} -> {v}"
                                    for k in keys for c, v in (results[k].get("claims") or {}).items())
        claims[cond] = dict(answers=len(keys), invented_before=sum(1 for k in keys if jobs[k].get("invented")),
                            invented_after=sum(1 for k in keys if results[k].get("invented")),
                            transitions=dict(moved.most_common()))
    plant = {kind: [results[key]["claims"].get(claim) for k, key, claim, _ in plants if k == kind]
             for kind in ("true", "false", "wrong_turn")}
    controls = {}
    for cond in CONDITIONS + ("plant_intrusion",):
        keys = sorted(k for k in new_controls if k.startswith(cond + "|"))
        controls[cond] = dict(n=len(keys), after=dict(collections.Counter(new_controls[k] for k in keys)),
                              before=dict(collections.Counter(old_controls.get(k, "-") for k in keys)) if cond in CONDITIONS else None)
    ok = lambda xs, want: round(sum(x == want for x in xs) / len(xs), 3) if xs else None
    true_ok = round(sum(x not in ("confirmed_invented", None, "pending") for x in plant["true"]) / len(plant["true"]), 3) if plant["true"] else None
    intr = controls["plant_intrusion"]["after"]
    intr_n = max(1, sum(intr.values()) - intr.get("same_reply", 0))
    # false (no turn at the minute) is decided by code since the dated-claim check (2026-11-03): a
    # check of the clock lookup; true and wrong_turn are the judge's.
    marks = dict(true_supported=true_ok, false_invented=ok(plant["false"], "confirmed_invented"),
                 wrong_turn_invented=ok(plant["wrong_turn"], "confirmed_invented"),
                 intrusion_clean_wins=round(intr.get("B", 0) / intr_n, 3) if planted else None)
    counts = dict(true_supported=len(plant["true"]), false_invented=len(plant["false"]),
                  wrong_turn_invented=len(plant["wrong_turn"]))
    return dict(policy=POLICY, history=history, claims=claims, controls=controls, plants=marks, plant_counts=counts,
                pass_marks=PASS, passed={k: v is not None and v >= PASS[k] for k, v in marks.items()})


def text(r):
    lines = [f"aux judge check {r['history']} ({r['policy']})", "", "claims (answers with an invented claim):"]
    for cond, c in r["claims"].items():
        lines.append(f"  {cond}: {c['invented_before']} -> {c['invented_after']} of {c['answers']} re-checked")
    lines.append("controls (A = condition, B = recent-only; B = memory hurt):")
    for cond, c in r["controls"].items():
        lines.append(f"  {cond}: {c['n']}  before {c['before']}  after {c['after']}")
    lines.append("plants vs pass marks:")
    for k, v in r["plants"].items():
        n = r.get("plant_counts", {}).get(k)
        lines.append(f"  {k}: {v} (mark {r['pass_marks'][k]}){f' over {n}' if n else ''} "
                     f"{'PASS' if r['passed'][k] else 'FAIL'}")
    return "\n".join(lines)


def sheet(jobs, results, old_controls, new_controls, contexts):
    """Every changed verdict, for the maintainer to read (private: under logs/)."""
    msg = {f"{u['condition']}|{u['ask']['id']}": u["ask"]["question"] for u, _, _ in contexts}
    lines = ["# Changed verdicts (claim check and controls, fixed prompts)", ""]
    for k in sorted(jobs):
        for claim, v in sorted((results[k].get("claims") or {}).items()):
            before = jobs[k]["claims"].get(claim)
            if before != v:
                lines.append(f"- claim `{k}`: {before} -> {v}: {claim}")
    lines.append("")
    for k in sorted(new_controls):
        if k in old_controls and old_controls[k] != new_controls[k]:
            lines.append(f"- control `{k}`: {old_controls[k]} -> {new_controls[k]}: {msg.get(k, '')[:200]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--grades", type=Path, required=True, help="Z2's S11 grades.json (the old verdicts)")
    parser.add_argument("--run", type=Path, required=True, help="Z2's run directory (s9-*, s10-*)")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("grades are private: keep them under logs/")
    ask = None
    if not args.dry_run:
        from experiments.lme.cloud_provider import load_selected_env
        from experiments.z2 import ledger_merge as M
        from experiments.z2.profiles import generator, metadata
        load_selected_env()
        ask = G.judge_caller(M.Calls(args.out / "aux-raw.jsonl"), generator(G.JUDGE).generate, metadata(G.JUDGE),
                             args.history)
    pending = run(args.history, args.grades, args.run, args.questions, args.out, ask, scale=args.scale, dry=args.dry_run)
    return 3 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
