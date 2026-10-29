"""Fact-replacement designs compared on the same pairs (Z2 F20; G60 follow-up).

Each design is a prompt, a JSON schema and a rule turning the answer into same / replaces /
both_true. All run through the production call (background model, temperature 0, constrained
JSON), one pair at a time; every answer is saved (durable, resumable).

  D0  pre-G60: same_information, both_can_be_true                          (main's prompt)
  D1  G60: + same_item                                                      (fix-pass 1b4b79b)
  D2  attribute: name the thing and the property each sentence states, then same_property and
      value_changed — replace only when the same property of the same thing got a new value
  D3  Graphiti-style: replace only when the newer sentence explicitly makes the earlier false
      (getzep/graphiti resolve_edge: "explicit evidence that the old fact stopped being true";
      never from recency, snapshots or a different instance)
  D4  D3 plus what the audit's judge saw: each fact's source turn excerpt and its position in the
      conversation (production has both at write time: the current turn, and the earlier fact's
      stored source row with its timestamp)

Sets: Z2 S8's 99 judged replacements (46 valid / 53 invalid; labels by Muse, ~14 read by eye,
some inconsistent) and the CODEX_V4 qualification set (99 real pairs, 12 planted with 5
distractors each). Acceptance (reconcile_eval.py): invalid kept <= 10/53, valid kept >= 38/46,
real >= 89/99, planted >= 10/12, distractors 0/60.

    uv run --no-sync python experiments/z2/supersession_designs.py --designs D0,D1,D2,D3 --out logs/z2-supersession
"""

import argparse
import collections
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "logs/z2-s0-reconcile-batch-2026-10-18-r1"))

D0_SYSTEM = (
    "You compare two memory facts said by the same speaker: an EARLIER one and a NEWER one. "
    "Answer two questions about what the sentences say:\n"
    "1. same_information: do both sentences state the same thing, with neither adding or "
    "changing any information?\n"
    "2. both_can_be_true: could both sentences be true at the same time, right now? Answer "
    "false only if the newer one says the earlier one is no longer so - a changed value, a "
    "cancelled or reversed decision, a stopped habit, a correction. Different details, extra "
    "detail, items of a list, a question, a suggestion, a possibility or a different direction "
    "of a relation can all be true together: answer true.\n"
    "Use only what the sentences say.")
D2_SYSTEM = (
    "You compare two memory facts said by the same speaker: an EARLIER one and a NEWER one.\n"
    "First write, for each sentence, the thing it is about and the property of that thing it "
    "states (for example thing: 'the indexer', property: 'port'; thing: 'the training run saved "
    "to v2.pt', property: 'final validation loss'). Then answer:\n"
    "- same_information: do both sentences state the same thing, neither adding nor changing "
    "anything?\n"
    "- same_property: do both state the SAME property of the SAME thing? A different run, file, "
    "error, batch, list, prompt or person is a different thing, even of the same kind. Extra "
    "detail about the same thing is the same property only if it restates the earlier value.\n"
    "- value_changed: if same_property, does the newer sentence give a different value for it - "
    "an updated number, a corrected statement, a reversed decision, a replaced definition? "
    "Restating, reaffirming or adding detail without changing the earlier value is not a change.\n"
    "Use only what the sentences say.")
D3_SYSTEM = (
    "You compare two memory facts said by the same speaker: an EARLIER one and a NEWER one.\n"
    "- same_information: do both sentences state the same thing, neither adding nor changing "
    "anything?\n"
    "- contradicts: does the NEWER sentence make the EARLIER one no longer true - it corrects it, "
    "updates the same value, reverses it, or says it stopped? Answer true only with explicit "
    "evidence in the sentences. Never infer it from the newer one being later. A different "
    "instance of the same kind (another run, file, error, batch, list or measurement of a "
    "different thing), extra detail, a restatement, a question, a suggestion or a possibility "
    "does not contradict.\n"
    "- reason: one short sentence.\n"
    "Use only what the sentences say.")

D4_SYSTEM = D3_SYSTEM.replace("Use only what the sentences say.",
    "Each fact comes with the conversation passage it was taken from and where in the conversation "
    "that passage is (turn number). Use the passages to tell whether the newer fact is about the same "
    "thing as the earlier one and whether it changes it, or is a separate occurrence, a restatement "
    "or a further detail. Judge only from the sentences and their passages.")


def schema(*fields, text=()):
    props = {f: {"type": "string"} for f in text} | {f: {"type": "boolean"} for f in fields}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


DESIGNS = {
    "D0": (D0_SYSTEM, schema("same_information", "both_can_be_true"), 80,
           lambda a: "same" if a["same_information"] else "both_true" if a["both_can_be_true"] else "replaces"),
    "D1": (None, None, 80,
           lambda a: "same" if a["same_information"] else
           "both_true" if a["both_can_be_true"] or not a.get("same_item") else "replaces"),
    "D2": (D2_SYSTEM, schema("same_information", "same_property", "value_changed",
                             text=("earlier_thing", "earlier_property", "newer_thing", "newer_property")), 220,
           lambda a: "same" if a["same_information"] else
           "replaces" if a["same_property"] and a["value_changed"] else "both_true"),
    "D3": (D3_SYSTEM, schema("same_information", "contradicts", text=("reason",)), 160,
           lambda a: "same" if a["same_information"] else "replaces" if a["contradicts"] else "both_true"),
    "D4": (D4_SYSTEM, schema("same_information", "contradicts", text=("reason",)), 160,
           lambda a: "same" if a["same_information"] else "replaces" if a["contradicts"] else "both_true"),
}
# D5: D4 with the reasoning written first — the constrained JSON follows the field order, so a reason
# placed after the booleans cannot inform them
D5_SYSTEM = D4_SYSTEM.replace("- same_information:", "- about: first, in one sentence, what each fact is about and whether the newer one "
    "changes the earlier one, repeats it, adds to it, or is a separate occurrence.\n- same_information:").replace("- reason: one short sentence.\n", "")
DESIGNS["D5"] = (D5_SYSTEM, {"type": "object", "additionalProperties": False,
                             "required": ["about", "same_information", "contradicts"],
                             "properties": {"about": {"type": "string"}, "same_information": {"type": "boolean"},
                                            "contradicts": {"type": "boolean"}}}, 200, DESIGNS["D4"][3])
CONTEXT = {"D4", "D5"}               # designs that read the source passages


def asker(design, out):
    """(earlier, newer) -> answer dict, cached per design in <out>/<design>-raw.jsonl."""
    from src.memory import facts
    from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name, json_schema
    from src.workers.completion_text import complete_text
    system, sch, cap, _ = DESIGNS[design]
    system, sch = system or facts.RECONCILE_SYSTEM, sch or facts.RECONCILE_SCHEMA
    path = out / f"{design}-raw.jsonl"
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            cache[(r["earlier"], r["newer"])] = r["answer"]

    def ask(earlier, newer, ctx=None):
        if (earlier, newer) in cache:
            return cache[(earlier, newer)]
        body = {"earlier": earlier, "newer": newer}
        if design in CONTEXT and ctx:
            body = {"earlier": {"fact": earlier, **ctx["earlier"]}, "newer": {"fact": newer, **ctx["newer"]}}
        completion = get_bg_client().chat.completions.create(
            model=get_bg_model_name(), temperature=0.0, max_tokens=cap, timeout=bg_timeout(cap),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
            response_format=json_schema("fact_relation", sch))
        answer = json.loads(complete_text(completion))
        cache[(earlier, newer)] = answer
        with path.open("a") as f:
            f.write(json.dumps(dict(earlier=earlier, newer=newer, answer=answer), ensure_ascii=False) + "\n")
        return answer
    return ask


def verdicts(design, pairs, out, workers=2):
    ask, rule = asker(design, out), DESIGNS[design][3]

    def one(p):
        if design in CONTEXT and len(p) < 3:
            return "both_true"                            # no passages for this set
        try:
            return rule(ask(*p))
        except Exception as exc:                          # noqa: BLE001 — production: a failure keeps both
            print(f"  {design}: call failed ({type(exc).__name__}) — both_true", flush=True)
            return "both_true"
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, pairs))


def window(source, fact, size=1200, step=400):
    """The source turn's passage that shares the most words with the fact (production: the cited span)."""
    import re
    text = source.get("text") or ""
    words = set(re.findall(r"[a-z0-9]{3,}", fact.lower()))
    starts = range(0, max(1, len(text) - size + step), step)
    best = max(starts, key=lambda a: len(words & set(re.findall(r"[a-z0-9]{3,}", text[a:a + size].lower()))))
    return dict(turn=source["turn"], passage=text[best:best + size])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--designs", default="D0,D1,D2,D3")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-qualification", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s8 = json.loads((ROOT / "logs/fix-pass/supersession-s8-pairs.json").read_text())
    passages = {}
    for h in ("B", "C"):
        items = json.loads((ROOT / f"logs/z2-run-main/s8-{h}/items.json").read_text())["items"]
        for i in items:
            if i["kind"] == "fact_supersession":
                src = json.loads(i["sources"]) if isinstance(i["sources"], str) else i["sources"]
                src = sorted(src, key=lambda s: s["turn"])
                if src:
                    passages[i["id"]] = (src[0], src[-1])
    by_id = {r["id"]: r for r in s8}
    for key, (a, b) in list(passages.items()):
        r = by_id.get(key)
        passages[key] = r and dict(earlier=window(a, r["earlier"]), newer=window(b, r["newer"]))
    qual = None
    if not args.skip_qualification:
        import qualify_batch as B
        it = json.loads((Path(B.L) / "dedupe-candidates.json").read_text())["items"]
        pairs = json.loads((Path(B.L) / "dedupe-label-pairs.json").read_text())
        cited = lambda x: dict(turn=x.get("turn"), passage=x.get("quote") or x["text"])   # the fact's cited span
        bare = lambda text: dict(turn=None, passage=text)                               # planted: no source turn
        real = [(it[p["old"]]["text"], it[p["new"]]["text"], B.G.REAL.get(n, "both_true"),
                 dict(earlier=cited(it[p["old"]]), newer=cited(it[p["new"]])))
                for n, p in enumerate(pairs) if n not in B.G.EXCLUDE]
        rng = random.Random(20261018)
        pool = sorted({p["old"] for p in pairs})
        planted, distract = [], []
        for old, new, gold, _note in B.G.PLANTED:
            planted.append((old, new, gold, dict(earlier=bare(old), newer=bare(new))))
            distract += [(it[o]["text"], new, dict(earlier=cited(it[o]), newer=bare(new))) for o in rng.sample(pool, 5)]
        qual = (real, planted, distract)
    ok = lambda got, gold: (got == gold or (gold == "not_same" and got != "same")
                            or (gold == "not_replaces" and got != "replaces"))
    for d in args.designs.split(","):
        got = verdicts(d, [(r["earlier"], r["newer"], passages.get(r["id"])) for r in s8], args.out)
        kept = lambda v: sum(1 for r, g in zip(s8, got) if r["valid"] == v and g == "replaces")
        line = f"{d}: S8 valid kept {kept(True)}/46, invalid kept {kept(False)}/53"
        if qual:
            real, planted, distract = qual
            gr = verdicts(d, [(e, n, c) for e, n, _, c in real], args.out)
            gp = verdicts(d, [(e, n, c) for e, n, _, c in planted], args.out)
            gd = verdicts(d, distract, args.out)
            line += (f" | real {sum(ok(g, r[2]) for g, r in zip(gr, real))}/{len(real)}"
                     f", planted {sum(ok(g, p[2]) for g, p in zip(gp, planted))}/{len(planted)}"
                     f", distractors replaced {sum(g == 'replaces' for g in gd)}/{len(gd)}")
        print(line, f"| verdicts {dict(collections.Counter(got))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
