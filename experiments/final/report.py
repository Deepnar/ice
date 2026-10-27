"""FINAL report (FINAL_PREREGISTRATION r2; FINAL_experiments.md §2.1): one table per dataset,
never pooled.

Per history, from grade-<h>/report.json's `scoring` (experiments/z2/scoring.py): M1 accuracy,
M2 supported current accuracy overall and on out-of-window questions, the hallucination rates,
prompt tokens and TUR; the primary tests' verdicts (§3), hallucination non-inferiority, M2 by
evidence distance, update accuracy, turn-cited evidence recall, context precision and — for B
and S9 — the fidelity audit; the judge's 1–5 as a descriptive column. Then: BEAM nugget means
per tier; LongMemEval-S (upstream yes/no + grade); run-to-run variance (S1 replayed twice, in
M1/M2); the cloud-answerer subset (does ICE keep its margin under a non-Meta answerer?); the
cross-family judge check (does Muse 1.3 grade Muse 1.2's answers more leniently?); and the
judge's agreement with the maintainer's blind sample.

    uv run --no-sync python experiments/final/report.py --run logs/final-run-main
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def condition_summary(report):
    """Per condition: weighted mean score and hallucination measures across question types."""
    out = {}
    for cond, kinds in (report or {}).get("by_condition_and_type", {}).items():
        cells = [c for c in kinds.values() if c.get("graded")]
        graded = sum(c["graded"] for c in cells)
        weighted = lambda field: (round(sum(c[field] * c["graded"] for c in cells if c.get(field) is not None)
                                        / max(1, sum(c["graded"] for c in cells if c.get(field) is not None)), 3)
                                  if any(c.get(field) is not None for c in cells) else None)
        out[cond] = dict(graded=graded, mean_score=weighted("mean_score"), superseded_rate=weighted("superseded_rate"),
                         invented_rate=weighted("invented_rate"),
                         abstention_accuracy=kinds.get("negative", {}).get("abstention_accuracy") or weighted("abstention_accuracy"),
                         attribution_score=(kinds.get("attribution") or {}).get("mean_score"))
    return out


def costs(read_dir, answers_dir):
    """Per condition: mean prompt tokens per answer (read passes) and mean answer seconds."""
    from experiments.z2.report import unit_index
    units = unit_index(read_dir) if Path(read_dir).exists() else {}
    secs = {}
    if (Path(answers_dir) / "answers.jsonl").exists():
        rows = {json.loads(l)["prompt_sha256"]: json.loads(l) for l in (Path(answers_dir) / "answers.jsonl").read_text().splitlines() if l.strip()}
        index = json.loads((Path(answers_dir) / "index.json").read_text())["index"]
        for key, sha in index.items():
            if sha in rows and rows[sha].get("seconds"):
                secs.setdefault(key.split("|", 1)[0], []).append(rows[sha]["seconds"])
    toks = {}
    for key, meta in units.items():
        if meta.get("prompt_tokens"):
            toks.setdefault(key.split("|", 1)[0], []).append(meta["prompt_tokens"])
    return {c: dict(prompt_tokens=round(statistics.fmean(toks[c])) if toks.get(c) else None,
                    answer_seconds=round(statistics.fmean(secs[c]), 1) if secs.get(c) else None)
            for c in sorted(set(toks) | set(secs))}


def beam(run):
    out = {}
    for tier in ("100K", "500K"):              # the 1M tier was dropped 2026-10-28
        cells = {}
        for path in sorted(Path(run).glob(f"grade-BEAM-{tier}-*/beam-report.json")):
            for cond, kinds in json.loads(path.read_text())["by_condition_and_type"].items():
                for c in kinds.values():
                    acc = cells.setdefault(cond, [0.0, 0])
                    if c["graded"] and c["mean_nugget"] is not None:
                        acc[0] += c["mean_nugget"] * c["graded"]
                        acc[1] += c["graded"]
        out[tier] = {c: dict(graded=n, mean_nugget=round(s / n, 4) if n else None) for c, (s, n) in cells.items()}
    return out


# Headline suites (pre-registration r2 §3, revised 2026-10-28): pooled first; each conversation is
# a deep dive under the aspect it was built to test. B and D were development histories (D tuned
# v3; Z2's evidence on B drives the fix pass), so they are pooled apart from the held-out chats.
# S8 is S1's style twin (same questions) — counted once, through S1.
SUITES = {
    "LSREP private, held out": ("A", "SW", "LAP"),
    "LSREP private, development (B's sealed questions, D)": ("B", "D"),
    "LSREP synthetic": ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9"),
}

HEADLINE = ("graded", "m1", "m2", "outdated", "invented", "confident_never_said", "attribution_wrong",
            "prompt_tokens", "tur", "tokens_per_correct")


def headline(scoring):
    """Per condition: the r2 headline numbers, with M2 on out-of-window questions beside them."""
    if not scoring:
        return {}
    out = {}
    for cond, row in scoring["by_condition"].items():
        out[cond] = {k: row.get(k) for k in HEADLINE}
        out[cond]["m2_out_of_window"] = scoring["by_window"][cond]["out"]["m2"]
        out[cond]["update_m2"] = scoring["update_accuracy"][cond]["m2"]
    return out


def history_table(run, h, read_dir, answers_dir):
    report = _load(run / f"grade-{h}" / "report.json") or {}
    sc = report.get("scoring")
    return dict(headline=headline(sc), primary_tests=(sc or {}).get("primary_tests"),
                noninferiority=(sc or {}).get("hallucination_noninferiority"),
                by_distance=(sc or {}).get("by_distance"), evidence_recall=(sc or {}).get("evidence_recall"),
                context_precision=(sc or {}).get("context_precision"),
                fidelity={c: f["state"] for c, f in ((sc or {}).get("fidelity") or {}).items()
                          if f["state"] != "not ablated here"} or None,
                write_cost=(sc or {}).get("write_cost"), window_rule=(sc or {}).get("window_rule"),
                questions=(sc or {}).get("questions"), out_of_window=(sc or {}).get("out_of_window"),
                descriptive_1_5=condition_summary(report), costs=costs(read_dir, answers_dir))


def cross_judge(run):
    """Muse 1.3 vs the cross-family judge on the subset contexts, pooled over the subset's
    histories (ask ids carry their history)."""
    from experiments.final.run_final import SUBSET
    from experiments.z2.scoring import judge_family_check
    graded = lambda path: ((_load(path) or {}).get("graded") or {})
    muse_main, other_main, muse_sub, other_sub = {}, {}, {}, {}
    for h in SUBSET:
        theirs_main = graded(run / f"crossjudge-{h}-main" / "grades.json")
        other_main.update(theirs_main)
        muse_main.update({k: v for k, v in graded(run / f"grade-{h}" / "grades.json").items() if k in theirs_main})
        other_sub.update(graded(run / f"crossjudge-{h}-subset" / "grades.json"))
        muse_sub.update(graded(run / f"subset-grade-{h}" / "grades.json"))
    if not (other_main and other_sub):
        return None
    return judge_family_check(muse_main, other_main, muse_sub, other_sub)


def locomo(run, exports=ROOT / "logs/final-locomo"):
    """LoCoMo (an extra table, prereg r2 §4): pooled over its 10 conversations, per ICE condition
    and per memory system — M1/M2 on categories 1–4 (comparable with other papers), confident
    answers to category 5 (never said), and LoCoMo's own token F1 on categories 1–4."""
    from experiments.locomo.adapter import native
    from experiments.z2 import scoring as S
    comparable = ("multi_hop", "temporal", "open_domain", "single_hop")
    run, rows, f1 = Path(run), {}, {}
    for grades in sorted(run.glob("grade-*LOCOMO-*/grades.json")):
        name = grades.parent.name                  # grade-LOCOMO-<id> | grade-baseline-<system>-LOCOMO-<id>
        sid = name.split("LOCOMO-", 1)[1]
        occ_path = Path(exports) / sid / "occurrences.json"
        occ = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}": o for o in json.loads(occ_path.read_text())}
        for key, g in json.loads(grades.read_text())["graded"].items():
            cond, ask = key.split("|", 1)
            if ask in occ:
                rows.setdefault(cond, []).append((occ[ask], S.outcome(g, occ[ask]["type"])))
        units = run / (f"local-read-LOCOMO-{sid}" if name.startswith("grade-LOCOMO") else name.replace("grade-", "local-", 1))
        answers = run / name.replace("grade-", "cloud-answers-", 1)
        if (answers / "index.json").exists():
            for cond, cells in native(units, answers, occ_path).items():
                acc = f1.setdefault(cond, [0.0, 0])
                for c in cells.values():
                    acc[0] += c["token_f1"] * c["n"]
                    acc[1] += c["n"]
    out = {}
    for cond, pairs in rows.items():
        main = S.summary([r for o, r in pairs if o["type"] in comparable])
        never = S.summary([r for o, r in pairs if o["type"] == "adversarial"])
        out[cond] = dict(graded=main["graded"], m1_categories_1_4=main["m1"], m2_categories_1_4=main["m2"],
                         confident_on_never_said=never["confident_never_said"],
                         token_f1_categories_1_4=round(f1[cond][0] / f1[cond][1], 4) if f1.get(cond, [0, 0])[1] else None)
    return out or None


def upper_bound(run):
    """Full history vs ICE's own `full` contexts, answered by the same model, BEAM's 128K tier:
    nugget means, and the paired difference (ICE minus full history) per question with a 95%
    interval over questions."""
    from experiments.z2.scoring import family_interval
    by_cond, paired = {}, {}
    for store in sorted(Path(run).glob("upper-grade-*/beam-grades.jsonl")):
        for line in store.read_text().splitlines():
            row = json.loads(line) if line.strip() else None
            if not row or row["status"] != "graded":
                continue
            cond, ask = row["key"].split("|", 1)
            by_cond.setdefault(cond, []).append(row["nugget_mean"])
            paired.setdefault(ask, {})[cond] = row["nugget_mean"]
    if not by_cond:
        return None
    diffs = {a: v["full"] - v["full_history"] for a, v in paired.items() if "full" in v and "full_history" in v}
    return dict(mean_nugget={c: round(statistics.fmean(v), 4) for c, v in by_cond.items()},
                questions=len(diffs), ice_minus_full_history=round(statistics.fmean(diffs.values()), 4) if diffs else None,
                ci95=family_interval(diffs) if len(diffs) > 1 else None)


def build(run):
    run = Path(run)
    histories = sorted(p.name.removeprefix("grade-") for p in run.glob("grade-*")
                       if (p / "report.json").exists())
    tables = {h: history_table(run, h, run / f"local-read-{h}", run / f"cloud-answers-{h}") for h in histories}
    variance = None
    if "S1" in tables and "S1-repeat" in tables:
        a, b = tables["S1"]["headline"], tables["S1-repeat"]["headline"]
        variance = {c: {m: round(b[c][m] - a[c][m], 4) for m in ("m1", "m2") if a[c][m] is not None and b[c][m] is not None}
                    for c in a if c in b}
    from experiments.z2.scoring import pool
    suites = {}
    for name, members in SUITES.items():
        parts = {h: (_load(run / f"grade-{h}" / "report.json") or {}).get("scoring") for h in members}
        parts = {h: sc for h, sc in parts.items() if sc}
        if parts:
            pooled = pool(parts)
            suites[name] = dict(histories=sorted(parts), missing=sorted(set(members) - set(parts)),
                                questions=pooled["questions"], out_of_window=pooled["out_of_window"],
                                headline=headline(dict(pooled, update_accuracy=pooled["update_accuracy"])),
                                primary_tests=pooled["primary_tests"], noninferiority=pooled["hallucination_noninferiority"],
                                by_distance=pooled["by_distance"])
    subset = {}
    for p in sorted(run.glob("subset-grade-*")):
        sc = (_load(p / "report.json") or {}).get("scoring")
        if sc:
            subset[p.name.removeprefix("subset-grade-")] = dict(headline=headline(sc), primary_tests=sc["primary_tests"])
    return dict(label="FINAL — suites pooled first, public benchmarks in their own scoring, then each conversation",
                suites=suites, histories=tables, beam=beam(run),
                longmemeval_s=_load(run / "grade-lmes" / "oracle-report.json"),
                run_to_run_variance_S1=variance, cloud_answerer_subset=subset or None,
                cross_family_judge=cross_judge(run), locomo=locomo(run), full_history_upper_bound=upper_bound(run),
                judge_vs_human=_load(run / "human-sample" / "agreement.json"))


def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def markdown(report):
    lines = [f"# {report['label']}", "",
             "M1 = correct; M2 = correct, current, nothing invented (declines never-said questions); "
             "out-of-window M2 = evidence beyond the long-context window. Percentages.", ""]

    def headline_table(head):
        out = ["| condition | graded | M1 | M2 | M2 out-of-window | M2 on updates | outdated | invented | "
               "confident never-said | prompt tokens | TUR |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, v in head.items():
            out.append(f"| {c} | {v['graded']} | {_pct(v['m1'])} | {_pct(v['m2'])} | {_pct(v['m2_out_of_window'])} | "
                       f"{_pct(v['update_m2'])} | {_pct(v['outdated'])} | {_pct(v['invented'])} | "
                       f"{_pct(v['confident_never_said'])} | {v['prompt_tokens']} | {v['tur']} |")
        return out

    def test_lines(tests, noninferior):
        out = []
        for name, test in (tests or {}).items():
            if test.get("pairs"):
                out.append(f"- **{name}**: {test['verdict']} — difference {_pct(test['diff'])} points, 95% CI "
                           f"{[_pct(x) for x in test['ci95'] or []]}, Holm p {test['p_holm']:.4g}, {test['pairs']} pairs "
                           f"(detectable ≈ {_pct(test['detectable'])})")
        for measure, test in (noninferior or {}).items():
            if test.get("pairs"):
                out.append(f"- non-inferiority, {measure}: {'shown' if test['non_inferior'] else 'NOT shown'} "
                           f"(upper bound {_pct((test['ci95'] or [None, None])[1])} points)")
        return out

    if report.get("suites"):
        lines += ["# 1. Suites (pooled)", ""]
        for name, sv in report["suites"].items():
            lines += [f"## {name}", "", f"{', '.join(sv['histories'])} — {sv['questions']} questions, "
                      f"{sv['out_of_window']} out of window" + (f"; not yet graded: {', '.join(sv['missing'])}" if sv["missing"] else ""), ""]
            lines += headline_table(sv["headline"]) + [""] + test_lines(sv["primary_tests"], sv["noninferiority"]) + [""]
        lines += ["# 2. Public benchmarks (each in its own scoring) — below", "", "# 3. Each conversation (deep dives)", ""]
    for h, t in report["histories"].items():
        lines += [f"## {h}", "", f"{t['questions']} questions, {t['out_of_window']} out of window ({t['window_rule']}).", "",
                  "| condition | graded | M1 | M2 | M2 out-of-window | M2 on updates | outdated | invented | "
                  "confident never-said | prompt tokens | TUR | 1–5 (descriptive) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, v in t["headline"].items():
            lines.append(f"| {c} | {v['graded']} | {_pct(v['m1'])} | {_pct(v['m2'])} | {_pct(v['m2_out_of_window'])} | "
                         f"{_pct(v['update_m2'])} | {_pct(v['outdated'])} | {_pct(v['invented'])} | "
                         f"{_pct(v['confident_never_said'])} | {v['prompt_tokens']} | {v['tur']} | "
                         f"{(t['descriptive_1_5'].get(c) or {}).get('mean_score')} |")
        lines.append("")
        for name, test in (t["primary_tests"] or {}).items():
            if test.get("pairs"):
                lines.append(f"- **{name}**: {test['verdict']} — difference {_pct(test['diff'])} points, 95% CI "
                             f"{[_pct(x) for x in test['ci95'] or []]}, Holm p {test['p_holm']:.4g}, {test['pairs']} pairs "
                             f"(detectable ≈ {_pct(test['detectable'])})")
        for measure, test in (t["noninferiority"] or {}).items():
            if test.get("pairs"):
                lines.append(f"- non-inferiority, {measure}: {'shown' if test['non_inferior'] else 'NOT shown'} "
                             f"(upper bound {_pct((test['ci95'] or [None, None])[1])} points)")
        if t["fidelity"]:
            lines.append("- fidelity: " + "; ".join(f"{c} {state}" for c, state in t["fidelity"].items()))
        lines.append("")
    for tier, conds in report["beam"].items():
        if conds:
            lines += [f"## BEAM {tier} (nugget protocol, non-official judge)", ""] + \
                     [f"- {c}: {v['mean_nugget']} over {v['graded']}" for c, v in conds.items()] + [""]
    if report["run_to_run_variance_S1"]:
        lines += ["## Run-to-run variance (S1 replayed twice; second minus first, points)", ""] + \
                 [f"- {c}: M1 {_pct(d.get('m1'))}, M2 {_pct(d.get('m2'))}" for c, d in report["run_to_run_variance_S1"].items()] + [""]
    if report["cloud_answerer_subset"]:
        lines += ["## Cloud-answerer subset (non-Meta answerer)", ""]
        for h, sub in report["cloud_answerer_subset"].items():
            for name, test in sub["primary_tests"].items():
                if test.get("pairs"):
                    lines.append(f"- {h} {name}: {test['verdict']} ({_pct(test['diff'])} points, {test['pairs']} pairs)")
        lines.append("")
    if report.get("locomo"):
        lines += ["## LoCoMo (extra table; histories fit one window; answer-key errors and lenient common judge noted)", "",
                  "| condition | graded (cat. 1–4) | M1 | M2 | confident on never-said (cat. 5) | token F1 |", "|---|---|---|---|---|---|"]
        for c, v in report["locomo"].items():
            lines.append(f"| {c} | {v['graded']} | {_pct(v['m1_categories_1_4'])} | {_pct(v['m2_categories_1_4'])} | "
                         f"{_pct(v['confident_on_never_said'])} | {v['token_f1_categories_1_4']} |")
        lines.append("")
    if report.get("full_history_upper_bound"):
        u = report["full_history_upper_bound"]
        lines += ["## Full-history upper bound (BEAM 128K tier, same model for both)", "",
                  f"- mean nugget: " + ", ".join(f"{c} {v}" for c, v in u["mean_nugget"].items()),
                  f"- ICE minus full history: {u['ice_minus_full_history']} over {u['questions']} questions, 95% CI {u['ci95']}", ""]
    if report["cross_family_judge"]:
        j = report["cross_family_judge"]
        lines += ["## Cross-family judge (does Muse 1.3 favour Muse 1.2's answers?)", "",
                  f"- {j['reading']}: family effect {_pct(j['family_effect'])} points, 95% CI {[_pct(x) for x in j['ci95'] or []]}",
                  f"- agreement on Muse answers {j['on_muse_answers']['agreement']} (kappa {j['on_muse_answers']['kappa']}); "
                  f"on other-family answers {j['on_other_family_answers']['agreement']} (kappa {j['on_other_family_answers']['kappa']})", ""]
    if report["judge_vs_human"]:
        j = report["judge_vs_human"]
        lines += ["## Judge vs the maintainer", "", f"- n {j['n']}, exact {j['exact']}, Cohen's kappa {j['cohen_kappa']}", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.run)
    (args.run / "final-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    (args.run / "final-report.md").write_text(markdown(report))
    print(args.run / "final-report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
