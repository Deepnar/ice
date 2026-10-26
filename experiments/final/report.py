"""FINAL report (FINAL_experiments.md §2.1; Z3 D11): one table per dataset, never pooled.

From a FINAL run folder: per history and condition, the mean score and the hallucination
measures Z2's report already computes (outdated-value use, invented claims, abstention accuracy
on never-said questions; synthetic attribution questions separately), prompt tokens per answer
and answer time; BEAM nugget means per tier; LongMemEval-S (upstream yes/no + grade);
run-to-run variance (S1 vs its second replay); the cloud-answerer subset beside the local
answers; and the judge's agreement with the maintainer's blind sample.

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
    for tier in ("100K", "500K", "1M"):
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


def build(run):
    run = Path(run)
    histories = sorted(p.name.removeprefix("grade-") for p in run.glob("grade-*")
                       if (p / "report.json").exists())
    tables = {h: dict(conditions=condition_summary(_load(run / f"grade-{h}" / "report.json")),
                      costs=costs(run / f"local-read-{h}", run / f"local-answers-{h}")) for h in histories}
    variance = None
    if "S1" in tables and "S1-repeat" in tables:
        a, b = tables["S1"]["conditions"], tables["S1-repeat"]["conditions"]
        variance = {c: round(b[c]["mean_score"] - a[c]["mean_score"], 3) for c in a
                    if c in b and a[c]["mean_score"] is not None and b[c]["mean_score"] is not None}
    cloud = {h.removeprefix("cloud-grade-"): condition_summary(_load(p / "report.json"))
             for h, p in ((p.name, p) for p in run.glob("cloud-grade-*")) if (p / "report.json").exists()}
    return dict(label="FINAL — one table per dataset; never pooled", histories=tables, beam=beam(run),
                longmemeval_s=_load(run / "grade-lmes" / "oracle-report.json"),
                run_to_run_variance_S1=variance, cloud_answerer_subset=cloud,
                judge_vs_human=_load(run / "human-sample" / "agreement.json"))


def markdown(report):
    lines = [f"# {report['label']}", ""]
    for h, t in report["histories"].items():
        lines += [f"## {h}", "", "| condition | graded | mean score | outdated used | invented | abstention acc. | prompt tokens | answer s |",
                  "|---|---|---|---|---|---|---|---|"]
        for c, v in t["conditions"].items():
            k = t["costs"].get(c, {})
            lines.append(f"| {c} | {v['graded']} | {v['mean_score']} | {v['superseded_rate']} | {v['invented_rate']} | "
                         f"{v['abstention_accuracy']} | {k.get('prompt_tokens')} | {k.get('answer_seconds')} |")
        lines.append("")
    for tier, conds in report["beam"].items():
        if conds:
            lines += [f"## BEAM {tier} (nugget protocol, non-official judge)", ""] + \
                     [f"- {c}: {v['mean_nugget']} over {v['graded']}" for c, v in conds.items()] + [""]
    if report["run_to_run_variance_S1"]:
        lines += ["## Run-to-run variance (S1 replayed twice; second minus first)", ""] + \
                 [f"- {c}: {d:+}" for c, d in report["run_to_run_variance_S1"].items()] + [""]
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
