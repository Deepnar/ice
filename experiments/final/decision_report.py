"""FINAL pre-registration revision 5: the memory decision, from what every read already records.

MDM  misses — questions needing memory whose prompt got none (gate.final_retrieve false: the decision
     after any rescue), overall and per question type (the benchmarks' own types included).
MDF  false alarms — no-memory controls where ICE fetched, with the memory tokens added and the read's
     seconds when it fetched vs when it did not (what a skipped read saves).
NMH  no-memory harm — on the controls, the share of a condition's replies the blind comparison judged
     worse than recent-only's (grade.py's `controls`; LoCoMo: locomo-controls.json).
POS  position contrast (paper 1's token-crossover table, extended) — per question, prompt tokens
     ICE − bare_vector and ICE − long_context in the first and last quarter of the history (paired,
     mean with a bootstrapped 95% interval over questions, 10,000 resamples, seed 0; the share where
     ICE is cheaper), and SCA (M2), MDM and MDF per quarter. Observational: query mix and state vary
     with position. Our own histories only — the benchmarks ask everything after the last turn.
"""

import random
import statistics

BASES = ("bare_vector", "long_context")
RESAMPLES, SEED = 10_000, 0


def _rate(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def _median(xs):
    return round(statistics.median(xs), 1) if xs else None


def decision(index, ice):
    """MDM (by type) and MDF (with tokens and seconds) for condition *ice* of one history."""
    rows = [r for k, r in index.items() if k.split("|", 1)[0] == ice and r.get("memory_given") is not None]
    asked = [r for r in rows if r["kind"] == "question"]
    controls = [r for r in rows if r["kind"] == "control"]
    by_type = {}
    for r in asked:
        by_type.setdefault(r["type"] or "untyped", []).append(not r["memory_given"])
    fetched = [r for r in controls if r["memory_given"]]
    return dict(
        mdm=_rate([not r["memory_given"] for r in asked]), questions=len(asked),
        mdm_by_type={t: dict(n=len(v), mdm=_rate(v)) for t, v in sorted(by_type.items())},
        mdf=_rate([r["memory_given"] for r in controls]), controls=len(controls),
        mdf_memory_tokens=_median([r["memory_tokens"] for r in fetched]),
        read_seconds_fetching=_median([r["read_seconds"] for r in rows if r["memory_given"] and r.get("read_seconds")]),
        read_seconds_skipping=_median([r["read_seconds"] for r in rows if not r["memory_given"] and r.get("read_seconds")]))


def harm(verdicts, baseline="recent_only"):
    """NMH per condition from {"cond|ask": "A"|"B"|"tie_or_split"|"same_reply"|"pending"}; A is the
    condition, B the no-memory reply."""
    by = {}
    for key, v in (verdicts or {}).items():
        if v != "pending":
            by.setdefault(key.split("|", 1)[0], []).append(v)
    return {c: dict(n=len(v), nmh=_rate([x == "B" for x in v]), better=_rate([x == "A" for x in v]))
            for c, v in sorted(by.items()) if c != baseline}


def interval(deltas, rng):
    means = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(RESAMPLES))
    return [round(means[int(0.025 * RESAMPLES)]), round(means[int(0.975 * RESAMPLES) - 1])]


def position(index, outcomes, turns, ice):
    """POS for one history: first vs last quarter of *turns* by each ask's checkpoint."""
    rng = random.Random(SEED)
    quarter = lambda cp: None if not cp or not turns else min(4, int(4 * (cp - 1) / turns) + 1)
    rows = {k.split("|", 1)[1]: r for k, r in index.items() if k.split("|", 1)[0] == ice}
    out = {}
    for q in (1, 2, 3, 4):
        here = {a: r for a, r in rows.items() if quarter(r.get("checkpoint_turn")) == q}
        cell = dict(questions=sum(r["kind"] == "question" for r in here.values()),
                    sca=_rate([outcomes[(ice, a)]["m2"] for a, r in here.items()
                               if r["kind"] == "question" and outcomes.get((ice, a))]),
                    mdm=_rate([not r["memory_given"] for r in here.values()
                               if r["kind"] == "question" and r.get("memory_given") is not None]),
                    mdf=_rate([r["memory_given"] for r in here.values()
                               if r["kind"] == "control" and r.get("memory_given") is not None]))
        if q in (1, 4):
            for base in BASES:
                deltas = [r["prompt_tokens"] - index[f"{base}|{a}"]["prompt_tokens"] for a, r in here.items()
                          if r["kind"] == "question" and r.get("prompt_tokens")
                          and (index.get(f"{base}|{a}") or {}).get("prompt_tokens")]
                cell[f"tokens_vs_{base}"] = dict(n=len(deltas), mean_delta=round(statistics.fmean(deltas)),
                                                 interval=interval(deltas, rng),
                                                 ice_cheaper=_rate([d < 0 for d in deltas])) if deltas else None
        out[f"Q{q}"] = cell
    return out


def history(index, grades, turns, ice):
    """Every revision-5 number for one of our histories."""
    from experiments.z2.scoring import outcome
    outcomes = {tuple(k.split("|", 1)): outcome(g) for k, g in (grades or {}).get("graded", {}).items()}
    return dict(decision=decision(index, ice), no_memory_harm=harm((grades or {}).get("controls")),
                position=position(index, outcomes, turns, ice))
