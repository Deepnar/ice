"""Scores and analyses for Z2 (spec D6e) and FINAL (FINAL_PREREGISTRATION r2 §2–§4), counted
from the judge's verdicts — no model is called here.

Per answer the judge gives correct / partial / incorrect / uncertain plus flags, and the claim
chain confirms each unsupported claim against the ledger and the originals (grade.py).
- **M1 accuracy:** the answer is `correct`; partial, incorrect and uncertain are not (uncertain
  is never credited, and its rate is reported).
- **M2 supported current accuracy:** correct, no outdated value, no confirmed-invented claim. A
  never-said question is correct only when the answer declines (the judge's EXPECT_ABSTENTION).
- **M3 cost:** prompt tokens per answer and per M1-correct answer, and TUR = M1-correct answers
  per 1,000 prompt tokens; the replay's wall seconds per turn beside it (model calls per turn
  are not in the replay trace, and the report says so).

**Distance** is the history's tokens between the farthest cited evidence turn and the question
(citations marked `current: false` — a superseded value's turn — are not evidence for the answer)
(`ledger.pair_tokens`, the counter the long-context condition packs with). **Out of window**
means the evidence lies before what `long_context`'s prompt held: exactly, from its packed
turns, where that condition was read; otherwise distance > LONG_CONTEXT_WINDOW. The split is
a property of the question, so every condition is scored on the same split.

**Tests** (§3, revised 2026-10-28): ICE as it runs (`full_default`) against long context and
bare vector — M1 and M2 on all questions, and M2 out of window where at least MIN_OUT_OF_WINDOW
pairs exist; McNemar's exact test on paired yes/no outcomes, the paired difference with a
bootstrap interval over question families (10,000 resamples, seed 0), Holm–Bonferroni across a
suite's tests, the detectable difference (80% power, 5% two-sided), and non-inferiority of the
hallucination rates against `bare_vector` (upper bound ≤ 2 points). `full` (decay neutral) is
only the ablations' reference.

**Paper 1's measures kept** (§4): update accuracy, context precision (from SPF), TUR (inside
M3), and the fidelity audit — per component: did it run, is its output faithful, does it reach
a prompt, does switching it off change any prompt and lower M2.
"""

import collections
import math
import random
import statistics

POLICY = "ice-scoring-1"
BINS = ((0, 32_000, "<32K"), (32_000, 128_000, "32K-128K"), (128_000, 512_000, "128K-512K"),
        (512_000, None, ">512K"))
LONG_CONTEXT_WINDOW = 256_000           # read_passes.LONG_CONTEXT_PROMPT_TOKENS (FINAL, 2026-10-28)
ICE = "full_default"                    # the frozen system as it runs — the headline (2026-10-28)
ABLATION_REFERENCE = "full"             # neutral decay: what leave-one-out conditions switch off from
MIN_OUT_OF_WINDOW = 100                 # P1 is tested only with at least this many paired questions
RESAMPLES, SEED = 10_000, 0
NONINFERIORITY_MARGIN = 0.02
ALPHA, POWER = 0.05, 0.80
ASSUMED_DISCORDANCE = 0.20              # for the detectable difference stated before a run
HALLUCINATION = ("invented", "confident_never_said", "outdated", "attribution_wrong")

# component → its background-audit kinds, its retrieval legs, its leave-one-out condition
COMPONENTS = {
    "bm25": dict(kinds=(), legs=("bm25", "bm25+vector")),
    "rerank": dict(kinds=(), legs=()),
    "codex": dict(kinds=("codex_fact", "fact_supersession", "entity_link"), legs=("facts",)),
    "summaries": dict(kinds=("turn_summary", "session_summary"), legs=()),
    "notes": dict(kinds=("conversation_note",), legs=()),
    "procedural": dict(kinds=("procedural",), legs=("procedural",)),
    "gate": dict(kinds=(), legs=()),
    "clusters": dict(kinds=("cluster",), legs=()),
    "temporal": dict(kinds=(), legs=()),
    "maintenance": dict(kinds=("entity_merge",), legs=()),
}


def family_of(ask_id):
    return ask_id.split("@")[0]


# ── one answer ───────────────────────────────────────────────────────────────

def outcome(g, ask_type=None):
    """The scored facts of one graded answer, or None when it was not graded."""
    if not g or g.get("status") != "graded":
        return None
    correct = g.get("grade") == "correct"
    outdated, invented = bool(g.get("superseded_used")), bool(g.get("invented"))
    never_said = bool(g.get("abstention_expected"))
    total = g.get("items_total")
    return dict(m1=correct, m2=correct and not outdated and not invented,
                uncertain=g.get("grade") == "uncertain", outdated=outdated, invented=invented,
                claims_pending=any(v == "pending" for v in (g.get("claims") or {}).values()),
                never_said=never_said, confident_never_said=never_said and not g.get("abstained"),
                attribution=ask_type == "attribution", attribution_wrong=ask_type == "attribution" and not correct,
                coverage=len(g.get("items_covered") or []) / total if total else None)


# ── distance ─────────────────────────────────────────────────────────────────

def turn_prefix(originals, count=None):
    """prefix[t] = tokens of turns 1..t (both roles)."""
    if count is None:
        from experiments.z2.ledger import pair_tokens
        count = lambda t: pair_tokens(originals, t)
    prefix = [0]
    for t in range(1, len(originals) + 1):
        prefix.append(prefix[-1] + count(t))
    return prefix


def distance(cited, checkpoint, prefix):
    """Tokens after the farthest cited turn, up to a question asked at checkpoint *checkpoint* —
    the store saved after that turn (replay's cp-<turn>), so turn *checkpoint* itself is
    history the question can rest on."""
    turns = [t for t in cited if isinstance(t, int) and 1 <= t <= checkpoint]
    if not turns:
        return None
    last = min(checkpoint, len(prefix) - 1)
    return prefix[last] - prefix[min(turns)]


def bin_of(d):
    if d is None:
        return "no cited turn"
    return next(label for lo, hi, label in BINS if d >= lo and (hi is None or d < hi))


def recent_turns(checkpoint, recent_tokens, prefix):
    """The turns a recent window of *recent_tokens* holds at the question (newest first)."""
    last = min(checkpoint, len(prefix) - 1)
    return {t for t in range(1, last + 1) if prefix[last] - prefix[t - 1] <= recent_tokens}


# ── statistics ───────────────────────────────────────────────────────────────

def mcnemar(b, c):
    """Exact two-sided p of b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def family_interval(per_family, resamples=RESAMPLES, seed=SEED):
    """95% interval of the mean of per-family values, resampling families."""
    values = list(per_family.values())
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples) - 1], 4)]


def holm(pvalues):
    """Holm–Bonferroni adjusted p per key."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def detectable(n, discordance=ASSUMED_DISCORDANCE, alpha=ALPHA, power=POWER):
    """Smallest paired difference McNemar detects with *n* pairs (Connor's approximation)."""
    if not n or discordance <= 0:
        return None
    za, zb = statistics.NormalDist().inv_cdf(1 - alpha / 2), statistics.NormalDist().inv_cdf(power)
    lo, hi = 1e-6, discordance - 1e-9
    for _ in range(80):
        mid = (lo + hi) / 2
        need = (za * math.sqrt(discordance) + zb * math.sqrt(discordance - mid ** 2)) ** 2 / mid ** 2
        lo, hi = (lo, mid) if need <= n else (mid, hi)
    return round(hi, 4) if hi < discordance - 1e-6 else None


def paired(outcomes, cond, base, measure, asks):
    """[(family, x, y)] for asks scored in both conditions."""
    out = []
    for ask in asks:
        x, y = outcomes.get((cond, ask)), outcomes.get((base, ask))
        if x is not None and y is not None:
            out.append((family_of(ask), bool(x[measure]), bool(y[measure])))
    return out


def compare(pairs):
    if not pairs:
        return dict(pairs=0)
    b = sum(1 for _, x, y in pairs if x and not y)
    c = sum(1 for _, x, y in pairs if y and not x)
    by_family = collections.defaultdict(list)
    for fam, x, y in pairs:
        by_family[fam].append(int(x) - int(y))
    per_family = {f: statistics.fmean(v) for f, v in by_family.items()}
    return dict(pairs=len(pairs), families=len(per_family), better=b, worse=c,
                diff=round(statistics.fmean(per_family.values()), 4), ci95=family_interval(per_family),
                p=mcnemar(b, c), detectable=detectable(len(pairs), max(ASSUMED_DISCORDANCE, (b + c) / len(pairs))))


def verdict(test, p_holm):
    ci = test.get("ci95")
    if ci and p_holm < ALPHA and ci[0] > 0:
        return "win"
    if ci and p_holm < ALPHA and ci[1] < 0:
        return "loss"
    return "no detectable difference"


# ── a history ────────────────────────────────────────────────────────────────

def rate(rows, field):
    rows = [r for r in rows if r.get(field) is not None]
    return round(sum(bool(r[field]) for r in rows) / len(rows), 4) if rows else None


def summary(rows, tokens=None):
    """M1, M2, the hallucination rates and M3 over scored rows (None rows are ungraded)."""
    done = [r for r in rows if r is not None]
    out = dict(n=len(rows), graded=len(done), ungraded=len(rows) - len(done),
               m1=rate(done, "m1"), m2=rate(done, "m2"), uncertain=rate(done, "uncertain"),
               outdated=rate(done, "outdated"), invented=rate(done, "invented"),
               claims_pending=sum(r["claims_pending"] for r in done),
               confident_never_said=rate([r for r in done if r["never_said"]], "confident_never_said"),
               attribution_wrong=rate([r for r in done if r["attribution"]], "attribution_wrong"),
               coverage=round(statistics.fmean(r["coverage"] for r in done if r["coverage"] is not None), 4)
               if any(r["coverage"] is not None for r in done) else None)
    if tokens is not None:
        spent = [t for t in tokens if t]
        correct = sum(r["m1"] for r in done)
        out.update(prompt_tokens=round(statistics.fmean(spent)) if spent else None,
                   tokens_per_correct=round(sum(spent) / correct) if spent and correct else None,
                   tur=round(correct / (sum(spent) / 1000), 4) if spent else None)
    return out


def score(history, graded, units, occurrences, prefix, *, semantic=None, prompts=None, audit=None,
          replay_walls=None, primary=ICE, bases=("long_context", "bare_vector"), targets=None):
    """Every §2–§4 number for one history.

    graded: grade.py's {"cond|ask": verdict}; units: report.unit_index rows (checkpoint, type,
    prompt_tokens, source_turns, recent_tokens, legs); occurrences: {ask: occurrence};
    prefix: turn_prefix of the originals; semantic: grade.py's semantic credit; prompts:
    answers index {"cond|ask": prompt sha}; audit: S8's bg-audit.json; replay_walls: the
    replay's per-turn wall times; targets: {ask: component} where questions were planned
    per component (FINAL's S9; read from the occurrences' `ask.component` when not given)."""
    outcomes, conditions, asks = {}, set(), set()
    for key, g in graded.items():
        cond, ask = key.split("|", 1)
        occ = occurrences.get(ask)
        if occ is None:
            continue
        conditions.add(cond)
        asks.add(ask)
        outcomes[(cond, ask)] = outcome(g, occ.get("type"))
    asks = sorted(asks)
    if targets is None:
        targets = {a: (occurrences[a].get("ask") or {}).get("component") for a in asks}
        targets = {a: c for a, c in targets.items() if c} or None

    # per question: distance, window side, whether its answer changed
    lc_from = {}
    for key, u in units.items():
        cond, ask = key.split("|", 1)
        if cond == "long_context" and u.get("source_turns"):
            lc_from[ask] = min(u["source_turns"])
    facts = {}
    for ask in asks:
        occ = occurrences[ask]
        cited = [c.get("turn") for c in occ["reference"].get("citations", []) if c.get("current", True)]
        d = distance(cited, occ["checkpoint_turn"], prefix)
        turns = [t for t in cited if isinstance(t, int) and t <= occ["checkpoint_turn"]]
        if not turns:
            side = None
        elif ask in lc_from:
            side = "out" if min(turns) < lc_from[ask] else "in"
        else:
            side = "out" if d > LONG_CONTEXT_WINDOW else "in"
        facts[ask] = dict(distance=d, bin=bin_of(d), window=side, turns=turns, family=family_of(ask),
                          type=occ.get("type"), updated=bool(occ["reference"].get("superseded")))

    def rows(cond, keep=lambda a: True):
        chosen = [a for a in asks if keep(a) and (cond, a) in outcomes]
        return [outcomes[(cond, a)] for a in chosen], [units.get(f"{cond}|{a}", {}).get("prompt_tokens") for a in chosen]

    conditions = sorted(conditions)
    by_condition = {c: summary(*rows(c)) for c in conditions}
    by_type = {c: {t: summary(rows(c, lambda a, t=t: facts[a]["type"] == t)[0])
                   for t in sorted({f["type"] for f in facts.values() if f["type"]})} for c in conditions}
    by_distance = {c: {label: summary(rows(c, lambda a, label=label: facts[a]["bin"] == label)[0])
                       for label in [b[2] for b in BINS] + ["no cited turn"]} for c in conditions}
    by_window = {c: {side: summary(rows(c, lambda a, side=side: facts[a]["window"] == side)[0])
                     for side in ("in", "out")} for c in conditions}
    update = {c: summary(rows(c, lambda a: facts[a]["updated"])[0]) for c in conditions}

    # turn-cited evidence recall: every cited turn is a fragment's source or inside the recent
    # window. Literal by design — a summary or fact that carries the content without citing every
    # turn is not credited here (TRAPS #32); context precision (judge-read) credits it.
    recall = {}
    for c in conditions:
        hits = []
        for a in asks:
            u = units.get(f"{c}|{a}")
            if u is None or not facts[a]["turns"] or (c, a) not in outcomes:
                continue
            held = set(u.get("source_turns") or []) | recent_turns(occurrences[a]["checkpoint_turn"],
                                                                    u.get("recent_tokens") or 0, prefix)
            hits.append(all(t in held for t in facts[a]["turns"]))
        recall[c] = dict(questions=len(hits), all_cited_turns_reached=round(sum(hits) / len(hits), 4) if hits else None,
                         note="turn citations only; compressed memory is credited by context_precision")

    # context precision (paper 1's SPF, length-free)
    precision = {}
    for key, r in (semantic or {}).items():
        c = precision.setdefault(key.split("|", 1)[0], collections.Counter())
        c["contexts"] += 1
        c["fragments"] += r.get("fragments_total") or 0
        c["carrying"] += r.get("fragments_carrying") or 0
        c["wrong"] += r.get("wrong") or 0
    precision = {c: dict(v, precision=round(v["carrying"] / v["fragments"], 4) if v["fragments"] else None,
                         wrong_share=round(v["wrong"] / v["fragments"], 4) if v["fragments"] else None)
                 for c, v in precision.items()}

    # primary tests (§3)
    tests = primary_tests(outcomes, asks, [a for a in asks if facts[a]["window"] == "out"], conditions, primary, bases)

    noninferior = {}
    if "bare_vector" in conditions and primary in conditions:
        for measure in HALLUCINATION:
            keep = {"confident_never_said": lambda a: facts[a]["type"] == "negative" or occurrences[a].get("expected_abstention"),
                    "attribution_wrong": lambda a: facts[a]["type"] == "attribution"}.get(measure, lambda a: True)
            t = compare(paired(outcomes, primary, "bare_vector", measure, [a for a in asks if keep(a)]))
            if t.get("pairs"):
                t["non_inferior"] = bool(t["ci95"] and t["ci95"][1] <= NONINFERIORITY_MARGIN)
            noninferior[measure] = t

    walls = sorted(replay_walls or [])
    gaps = sorted(b - a for a, b in zip(walls, walls[1:]) if 0 <= b - a < 3600)
    write_cost = dict(turns=len(walls), median_seconds_per_turn=round(statistics.median(gaps), 2) if gaps else None,
                      model_calls_per_turn="not in the replay trace")

    per_question = {a: dict(family=facts[a]["family"], window=facts[a]["window"], bin=facts[a]["bin"],
                            type=facts[a]["type"], updated=facts[a]["updated"],
                            never_said=bool(occurrences[a].get("expected_abstention")),
                            outcomes={c: dict(outcomes[(c, a)], prompt_tokens=units.get(f"{c}|{a}", {}).get("prompt_tokens"))
                                      for c in conditions if outcomes.get((c, a)) is not None})
                    for a in asks}
    return dict(policy=POLICY, history=history, primary=primary, bases=list(bases), per_question=per_question,
                questions=len(asks), out_of_window=sum(f["window"] == "out" for f in facts.values()),
                window_rule="long_context's packed turns" if lc_from else f"distance > {LONG_CONTEXT_WINDOW} tokens",
                by_condition=by_condition, by_type=by_type, by_distance=by_distance, by_window=by_window,
                update_accuracy=update, evidence_recall=recall, context_precision=precision,
                primary_tests=tests, hallucination_noninferiority=noninferior,
                fidelity=fidelity(outcomes, asks, conditions, units, prompts, audit, ABLATION_REFERENCE, targets),
                write_cost=write_cost)


def primary_tests(outcomes, asks, out_asks, conditions, primary=ICE, bases=("long_context", "bare_vector")):
    """Pre-registration r2 §3 (revised 2026-10-28): against each base, M1 and M2 on all questions,
    and M2 on the questions whose evidence lies beyond the long-context window — that one only
    with MIN_OUT_OF_WINDOW paired questions or more, else reported as too few. Holm across the
    tests run."""
    tests = {}
    for base in [b for b in bases if b in conditions and primary in conditions]:
        tests[f"M1 all: {primary} vs {base}"] = compare(paired(outcomes, primary, base, "m1", asks))
        tests[f"M2 all: {primary} vs {base}"] = compare(paired(outcomes, primary, base, "m2", asks))
        out = compare(paired(outcomes, primary, base, "m2", out_asks))
        if out.get("pairs", 0) < MIN_OUT_OF_WINDOW:
            out = dict(pairs=out.get("pairs", 0), verdict=f"not tested: fewer than {MIN_OUT_OF_WINDOW} out-of-window pairs")
        tests[f"P1 M2 out-of-window: {primary} vs {base}"] = out
    adjusted = holm({k: t["p"] for k, t in tests.items() if t.get("p") is not None})
    for k, t in tests.items():
        if t.get("p") is not None:
            t.update(p_holm=adjusted[k], verdict=verdict(t, adjusted[k]))
    return tests


def pool(parts, primary=ICE, bases=("long_context", "bare_vector")):
    """One suite's headline from several histories (pre-registration r2 §3, revised 2026-10-28):
    *parts* maps history -> its score()'s output. Questions keep their history (ids and families
    are prefixed with it, so the interval still resamples whole families); M1, M2, the window
    split, distance bins, the primary tests (Holm across this suite's tests) and hallucination
    non-inferiority are computed on the pooled questions."""
    outcomes, facts = {}, {}
    for history, sc in parts.items():
        for ask, q in ((sc or {}).get("per_question") or {}).items():
            key = f"{history}|{ask}"
            facts[key] = q
            for cond, o in q["outcomes"].items():
                outcomes[(cond, key)] = o
    asks = sorted(facts)
    conditions = sorted({c for c, _ in outcomes})

    def rows(cond, keep=lambda a: True):
        chosen = [a for a in asks if keep(a) and (cond, a) in outcomes]
        return [outcomes[(cond, a)] for a in chosen], [outcomes[(cond, a)].get("prompt_tokens") for a in chosen]

    out_asks = [a for a in asks if facts[a]["window"] == "out"]
    tests = primary_tests(outcomes, asks, out_asks, conditions, primary, bases)
    noninferior = {}
    if "bare_vector" in conditions and primary in conditions:
        for measure in HALLUCINATION:
            keep = {"confident_never_said": lambda a: facts[a]["never_said"],
                    "attribution_wrong": lambda a: facts[a]["type"] == "attribution"}.get(measure, lambda a: True)
            t = compare(paired(outcomes, primary, "bare_vector", measure, [a for a in asks if keep(a)]))
            if t.get("pairs"):
                t["non_inferior"] = bool(t["ci95"] and t["ci95"][1] <= NONINFERIORITY_MARGIN)
            noninferior[measure] = t
    return dict(policy=POLICY, histories=sorted(parts), questions=len(asks), out_of_window=len(out_asks),
                by_condition={c: summary(*rows(c)) for c in conditions},
                by_window={c: {side: summary(rows(c, lambda a, side=side: facts[a]["window"] == side)[0])
                               for side in ("in", "out")} for c in conditions},
                by_distance={c: {label: summary(rows(c, lambda a, label=label: facts[a]["bin"] == label)[0])
                                 for label in [b[2] for b in BINS] + ["no cited turn"]} for c in conditions},
                update_accuracy={c: summary(rows(c, lambda a: facts[a]["updated"])[0]) for c in conditions},
                primary_tests=tests, hallucination_noninferiority=noninferior)


def fidelity(outcomes, asks, conditions, units, prompts, audit, primary="full", targets=None):
    """Per component: did it run, is its output faithful, does it reach a prompt, does
    switching it off change any prompt, and the M2 change on its questions."""
    kinds = (audit or {}).get("kinds", {})
    out = {}
    for name, spec in COMPONENTS.items():
        loo = f"loo_{name}"
        produced = {k: kinds[k].get("items") for k in spec["kinds"] if k in kinds}
        faithful = {k: (kinds[k].get("criteria", {}).get("faithful") or {}).get("yes_rate") for k in spec["kinds"] if k in kinds}
        reach = None
        if spec["legs"]:
            mine = [u for k, u in units.items() if k.startswith(f"{primary}|")]
            reach = round(sum(1 for u in mine if set(u.get("legs") or ()) & set(spec["legs"])) / len(mine), 4) if mine else None
        changed = None
        if prompts and loo in conditions:
            both = [a for a in asks if f"{primary}|{a}" in prompts and f"{loo}|{a}" in prompts]
            changed = sum(prompts[f"{primary}|{a}"] != prompts[f"{loo}|{a}"] for a in both) if both else None
        planned = [a for a in asks if (targets or {}).get(a) == name] if targets else asks
        effect = compare(paired(outcomes, primary, loo, "m2", planned)) if loo in conditions else None
        if loo not in conditions:
            state = "not ablated here"
        elif changed == 0 or (produced and not any(produced.values())):
            state = "unexercised"
        elif effect and effect.get("ci95") and effect["ci95"][0] > 0:
            state = "contributing"
        elif effect and effect.get("ci95") and effect["ci95"][1] < 0:
            state = "defective (switching it off raises M2)"
        else:
            state = "no measurable effect"
        out[name] = dict(items=produced or None, faithful=faithful or None, prompt_reach=reach,
                         prompts_changed_when_off=changed, questions="planned for it" if targets else "all",
                         m2_full_minus_off=effect, state=state)
    return out


# ── the cross-family judge check (Z3 D13) ────────────────────────────────────

def kappa(pairs):
    """Cohen's kappa of two yes/no raters."""
    n = len(pairs)
    if not n:
        return None
    agree = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return round((agree - expected) / (1 - expected), 4) if expected < 1 else None


def judge_family_check(muse_on_main, other_on_main, muse_on_subset, other_on_subset,
                       resamples=RESAMPLES, seed=SEED):
    """Does the Muse judge favour answers written by its own family? Each argument maps
    "cond|ask" to grade.py's verdict. Leniency = Muse's M1 minus the other judge's M1 on the
    same answers; the family effect = leniency on Muse 1.2's answers minus leniency on the
    non-Meta answerer's, with a 95% interval over question families."""
    def scored(muse, other):
        out = {}
        for key in set(muse) & set(other):
            a, b = outcome(muse[key]), outcome(other[key])
            if a is not None and b is not None:
                out[key] = (a["m1"], b["m1"])
        return out

    main, sub = scored(muse_on_main, other_on_main), scored(muse_on_subset, other_on_subset)

    def side(pairs):
        return dict(answers=len(pairs), muse_m1=rate([dict(x=a) for a, _ in pairs.values()], "x"),
                    other_m1=rate([dict(x=b) for _, b in pairs.values()], "x"),
                    agreement=round(sum(a == b for a, b in pairs.values()) / len(pairs), 4) if pairs else None,
                    kappa=kappa(list(pairs.values())))

    families = collections.defaultdict(lambda: ([], []))
    for i, pairs in enumerate((main, sub)):
        for key, (a, b) in pairs.items():
            families[family_of(key.split("|", 1)[1])][i].append(int(a) - int(b))
    names = sorted(families)

    def effect(chosen):
        m = [v for f in chosen for v in families[f][0]]
        s = [v for f in chosen for v in families[f][1]]
        return statistics.fmean(m) - statistics.fmean(s) if m and s else None

    point = effect(names)
    interval = None
    if point is not None and len(names) >= 2:
        rng = random.Random(seed)
        draws = sorted(e for e in (effect(rng.choices(names, k=len(names))) for _ in range(resamples)) if e is not None)
        if draws:
            interval = [round(draws[int(0.025 * len(draws))], 4), round(draws[int(0.975 * len(draws)) - 1], 4)]
    return dict(on_muse_answers=side(main), on_other_family_answers=side(sub),
                family_effect=round(point, 4) if point is not None else None, ci95=interval,
                reading="Muse grades its own family's answers more leniently than the other judge does"
                if interval and interval[0] > 0 else "no detectable family leniency" if interval else "too few answers")
