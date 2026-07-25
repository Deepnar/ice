#!/usr/bin/env python3
"""Stage 4 — label the corpus with two independent local models, then reconcile.

Rewrites ``legacy/promt_labeling/VLLM_label_dataset.py`` (single labeler,
``instructor`` retry loop, v1 taxonomy) and folds in ``compare_labeling.py``'s
diff logic as the agreement step.

The methodology, and why each piece is there:

* **Two labelers from DIFFERENT model families**, run sequentially (24 GB holds
  one at a time), each labeling from scratch and blind to the other — and blind
  to the v1 labels, which are never shown and never mapped forward. Independence
  is the entire quality signal: two Qwen variants agreeing tells you a model
  agrees with itself.
* **Agreement → keep. Disagreement → a third family breaks the tie. Still split
  → a human decides.** The human queue is the point of the exercise, not a
  failure mode: it concentrates the user's limited review time on exactly the
  rows where competent models genuinely disagree.
* **Temporal_Recall gets free weak supervision.** T2's deterministic detector
  (``src/retrieval/timescope.py``) runs over every row; a resolvable temporal
  expression in a recall-shaped prompt is a positive with no model in the loop.
  This is the only label that starts from zero positives and can be seeded
  without fabrication. The detector adds positives; it never removes them, and it
  never sets a time window here (that stays a retrieval-time concern, D7).

Usage:
    # one labeler over the corpus (resumable; re-run after a crash)
    uv run python scripts/classifier/pipeline/label.py --labeler A
    uv run python scripts/classifier/pipeline/label.py --labeler B

    # reconcile, tiebreak the disagreements, emit the human queue + audit sample
    uv run python scripts/classifier/pipeline/label.py --merge
    uv run python scripts/classifier/pipeline/label.py --labeler C --tiebreak-only
    uv run python scripts/classifier/pipeline/label.py --merge
"""

import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter

import rubric
from common import (AUDIT_SAMPLE, CORPUS_RAW, CORPUS_SYNTH, ICEDEV_STITCHED, LABELS_A,
                    LABELS_B, LABELS_FINAL, LABELS_TIEBREAK, REVIEW_QUEUE,
                    JsonlAppender, completed_ids, ensure_data_dir, read_jsonl,
                    write_jsonl)
from serving import LocalServer, is_up, served_model

from src.classifier.schema import (CONTEXT_RELIANCE, INTENT, TEMPORAL_RECALL,
                                   TOPIC, load_schema)

SEED = 42
AUDIT_FRACTION = 0.05

# Degenerate-output gate (see is_degenerate): after this many rows, abort if more
# than this fraction look like token soup.
GATE_SAMPLE = 15
GATE_MAX_DEGENERATE = 0.3

# Three DIFFERENT families — mistral / gemma / qwen. The family split is the
# methodology, not a preference: agreement between two members of one family
# measures a model against itself, and the agreement rate is the number the whole
# labeling design rests on.
LABELERS = {
    "A": {"model": "qwen3-14b", "out": LABELS_A},
    "B": {"model": "gemma-4-26b-a4b", "out": LABELS_B},
    "C": {"model": "mistral-small-24b", "out": LABELS_TIEBREAK},
}


# ── corpus ──────────────────────────────────────────────────────────────────

def load_corpus(paths, limit: int = 0, stratified: bool = True) -> list:
    rows, seen = [], set()
    for path in paths:
        for row in read_jsonl(path):
            if row.get("id") and row["id"] not in seen:
                seen.add(row["id"])
                rows.append(row)
    if limit and limit < len(rows):
        if stratified:
            # A dry run must look like the corpus, not like its first N rows —
            # otherwise the throughput and quality numbers it produces are lies.
            by_source = {}
            for row in rows:
                by_source.setdefault(row.get("source", "?"), []).append(row)
            rng = random.Random(SEED)
            picked, per = [], max(1, limit // max(1, len(by_source)))
            for source_rows in by_source.values():
                rng.shuffle(source_rows)
                picked += source_rows[:per]
            rng.shuffle(picked)
            rows = picked[:limit]
        else:
            rows = rows[:limit]
    return rows


# ── one labeling pass ───────────────────────────────────────────────────────

async def _label_row(client, model_id, system_prompt, schema_json, row,
                     response_mode, semaphore, out, failed, progress):
    from openai import APIError

    async with semaphore:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": rubric.build_user_message(row)}]
        kwargs = {"model": model_id, "messages": messages, "temperature": 0.0,
                  "max_tokens": 700, "seed": SEED}
        if response_mode == "response_format":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "labels", "schema": schema_json},
            }
        else:
            kwargs["extra_body"] = {"guided_json": schema_json}

        try:
            resp = await client.chat.completions.create(**kwargs)
            payload = json.loads(resp.choices[0].message.content)
        except (APIError, json.JSONDecodeError, KeyError, IndexError) as exc:
            failed.write({"id": row["id"], "error": str(exc)[:300]})
            progress()
            return

        reasoning = payload.get("reasoning", "")
        out.write({
            "id": row["id"],
            "source": row.get("source"),
            # The prompt travels WITH its labels. Costs ~40 MB over the corpus and
            # makes every output file reviewable on its own — spot-checking a
            # labeler shouldn't require re-joining against corpus_raw by id.
            "text": row.get("text", ""),
            "context_text": row.get("context_text"),
            "labels": {
                # Constrained decoding can emit the same enum value twice; dedupe
                # while preserving schema order.
                TOPIC: list(dict.fromkeys(payload.get("topic", []))),
                INTENT: list(dict.fromkeys(payload.get("intent", []))),
                CONTEXT_RELIANCE: list(dict.fromkeys(payload.get("context_reliance", []))),
            },
            "reasoning": reasoning[:1500],
            "labeler": model_id,
        })
        progress(degenerate=is_degenerate(reasoning))


def is_degenerate(reasoning: str) -> bool:
    """Does this reasoning look like token soup rather than an argument?

    Constrained decoding guarantees *parseable* output, not *correct* output. A
    broken quantization happily emits schema-valid JSON with real enum values and
    a reasoning field of `ìľ¼ëĤĺ...jumjumjum...` — which looks completely fine in
    the output file and is worthless as a label. (Observed: one Mistral requant
    did exactly this for 18/18 rows.) Two cheap signals catch it:

      * heavy non-ASCII content in what should be English reasoning
      * a token repeated many times in a row (the classic degenerate-decode loop)
    """
    if not reasoning or len(reasoning) < 40:
        return True
    non_ascii = sum(1 for ch in reasoning if ord(ch) > 127)
    if non_ascii / len(reasoning) > 0.08:
        return True
    words = reasoning.split()
    # A degenerate decode often drops spaces entirely and returns one long blob.
    if words and max(len(w) for w in words) > 60:
        return True
    if len(words) >= 12:
        run = best = 1
        for a, b in zip(words, words[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        if best >= 4:
            return True
        if len(set(words)) / len(words) < 0.35:
            return True
    return False


async def _detect_response_mode(client, model_id, schema_json) -> str:
    """Ask the server for constrained JSON the OpenAI way; fall back to the
    guided_json extension. Probing once beats guessing per backend version."""
    for mode in ("response_format", "guided_json"):
        try:
            kwargs = {"model": model_id, "max_tokens": 40, "temperature": 0.0,
                      "messages": [{"role": "user", "content": "Label: hi"}]}
            if mode == "response_format":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "labels", "schema": schema_json}}
            else:
                kwargs["extra_body"] = {"guided_json": schema_json}
            await client.chat.completions.create(**kwargs)
            print(f"[label] constrained decoding via {mode}")
            return mode
        except Exception as exc:
            print(f"[label] {mode} unavailable: {str(exc)[:120]}")
    raise RuntimeError("server supports neither response_format nor guided_json — "
                       "constrained decoding is required (it replaces the v1 retry loop)")


async def run_labeler(rows, out_path, base_url, model_id, concurrency):
    from openai import AsyncOpenAI

    schema = load_schema()
    system_prompt = rubric.build_system_prompt(schema)
    schema_json = rubric.response_json_schema(schema)
    client = AsyncOpenAI(base_url=base_url, api_key="local", timeout=600.0,
                         max_retries=2)

    done = completed_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"[label] {len(done)} already labeled, {len(todo)} to go "
          f"(rubric {len(system_prompt.split())} words, shared across all rows)")
    if not todo:
        return

    mode = await _detect_response_mode(client, model_id, schema_json)
    semaphore = asyncio.Semaphore(concurrency)
    started, counter = time.time(), {"n": 0, "degenerate": 0}
    aborted = asyncio.Event()

    def progress(degenerate: bool = False):
        counter["n"] += 1
        counter["degenerate"] += int(degenerate)
        n, bad = counter["n"], counter["degenerate"]
        # Sanity gate: a broken quantization emits schema-valid JSON with
        # gibberish reasoning, which is invisible in the output file and useless
        # as a label. Catch it in the first minute, not after five hours.
        if n >= GATE_SAMPLE and bad / n > GATE_MAX_DEGENERATE and not aborted.is_set():
            aborted.set()
            print(f"\n!!! [label] ABORTING: {bad}/{n} rows have degenerate "
                  f"reasoning — this model is producing token soup behind valid "
                  f"JSON. Constrained decoding guarantees parseable output, not "
                  f"correct output. Pick a different quantization.\n", flush=True)
        if n % 25 == 0 or n == len(todo):
            rate = n / max(1e-9, time.time() - started)
            eta = (len(todo) - n) / max(rate, 1e-9)
            suffix = f"  [{bad} degenerate]" if bad else ""
            print(f"  {n}/{len(todo)}  {rate:.2f} rows/s  ETA {eta / 60:.0f} min{suffix}",
                  flush=True)

    failed_path = out_path.replace(".jsonl", "_failed.jsonl")
    with JsonlAppender(out_path) as out, JsonlAppender(failed_path) as failed:
        tasks = [asyncio.create_task(
            _label_row(client, model_id, system_prompt, schema_json, row, mode,
                       semaphore, out, failed, progress))
            for row in todo]
        watcher = asyncio.create_task(aborted.wait())
        done_tasks, pending = await asyncio.wait(
            [*tasks, watcher], return_when=asyncio.FIRST_COMPLETED)
        while pending and not aborted.is_set():
            done_tasks, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
        if aborted.is_set():
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        watcher.cancel()

    elapsed = time.time() - started
    print(f"[label] {counter['n']} rows in {elapsed / 60:.1f} min "
          f"({counter['n'] / max(elapsed, 1e-9):.2f} rows/s)")
    if aborted.is_set():
        raise SystemExit(f"labeling aborted — {counter['degenerate']}/{counter['n']} "
                         f"degenerate rows from {model_id}")


# ── reconciliation ──────────────────────────────────────────────────────────

def _sets(entry):
    labels = entry.get("labels", {})
    return ({t for t in labels.get(TOPIC, [])},
            {i for i in labels.get(INTENT, [])},
            {c for c in labels.get(CONTEXT_RELIANCE, [])})


def _agreement(a, b):
    """Per-head agreement between two labelers.

    Context reliance demands an EXACT match: four binary signals, and they are
    the labels this whole retrain exists to get right. Topic and intent are
    inherently fuzzy multi-label sets where two competent labelers routinely
    pick 2-of-3 the same, so requiring exact equality there would bury the user
    under thousands of reviews that don't change the trained model.
    """
    ta, ia, ca = _sets(a)
    tb, ib, cb = _sets(b)
    return {
        TOPIC: bool(ta & tb),
        INTENT: bool(ia & ib),
        CONTEXT_RELIANCE: ca == cb,
    }


def _merge_agreeing(a, b):
    ta, ia, ca = _sets(a)
    tb, ib, cb = _sets(b)
    return {
        # Intersection, not union: a label both models saw is a label worth
        # training on. Union imports each model's false positives into the target.
        TOPIC: sorted(ta & tb) or sorted(ta),
        INTENT: sorted(ia & ib) or sorted(ia),
        CONTEXT_RELIANCE: sorted(ca),
    }


def _majority(head, a, b, c):
    """Two-of-three on one head; None when all three differ."""
    idx = {TOPIC: 0, INTENT: 1, CONTEXT_RELIANCE: 2}[head]
    sa, sb, sc = _sets(a)[idx], _sets(b)[idx], _sets(c)[idx]
    if head == CONTEXT_RELIANCE:
        if sc == sa or sc == sb:
            return sorted(sc)
        return None
    for x, y in ((sa, sc), (sb, sc), (sa, sb)):
        if x & y:
            return sorted(x & y)
    return None


def temporal_weak_supervision(rows_by_id, final_rows) -> int:
    """T2's detector over the corpus: every hit is a Temporal_Recall positive.

    Called with default gating arguments — no p_ltm, no reference signal — so it
    only fires on prompts that carry a resolvable time expression AND are shaped
    like a question. High precision is what makes this usable as a label without
    a model in the loop.
    """
    from src.retrieval.timescope import detect_timescope

    added = 0
    for entry in final_rows:
        row = rows_by_id.get(entry["id"])
        if not row:
            continue
        scope = detect_timescope(row.get("text", ""))
        if scope.mode == "current":
            continue
        ctx = entry["labels"][CONTEXT_RELIANCE]
        if TEMPORAL_RECALL not in ctx:
            ctx.append(TEMPORAL_RECALL)
            entry.setdefault("provenance", {})["temporal_detector"] = scope.mode
            added += 1
        else:
            entry.setdefault("provenance", {})["temporal_detector"] = scope.mode
    return added


def merge(corpus_paths, verbose=True):
    schema = load_schema()
    rows_by_id = {r["id"]: r for r in load_corpus(corpus_paths)}
    a_by_id = {e["id"]: e for e in read_jsonl(LABELS_A)}
    b_by_id = {e["id"]: e for e in read_jsonl(LABELS_B)}
    c_by_id = {e["id"]: e for e in read_jsonl(LABELS_TIEBREAK)}

    both = [i for i in a_by_id if i in b_by_id]
    print(f"[merge] A={len(a_by_id)} B={len(b_by_id)} C={len(c_by_id)} "
          f"overlap={len(both)}")

    final, needs_tiebreak, needs_human = [], [], []
    head_disagreements = Counter()

    for row_id in both:
        a, b = a_by_id[row_id], b_by_id[row_id]
        agree = _agreement(a, b)
        for head, ok in agree.items():
            if not ok:
                head_disagreements[head] += 1

        if all(agree.values()):
            final.append({"id": row_id, "source": a.get("source"),
                          "labels": _merge_agreeing(a, b),
                          "agreement": "both"})
            continue

        c = c_by_id.get(row_id)
        if c is None:
            needs_tiebreak.append(row_id)
            continue

        resolved, unresolved = {}, []
        for head, ok in agree.items():
            if ok:
                resolved[head] = _merge_agreeing(a, b)[head]
            else:
                won = _majority(head, a, b, c)
                if won is None:
                    unresolved.append(head)
                else:
                    resolved[head] = won
        if unresolved:
            needs_human.append({
                "id": row_id,
                "text": rows_by_id.get(row_id, {}).get("text", ""),
                "context_text": rows_by_id.get(row_id, {}).get("context_text"),
                "source": a.get("source"),
                "disputed_heads": unresolved,
                "candidates": {"A": a["labels"], "B": b["labels"], "C": c["labels"]},
                "reasoning": {"A": a.get("reasoning", "")[:600],
                              "B": b.get("reasoning", "")[:600],
                              "C": c.get("reasoning", "")[:600]},
                "resolved_heads": resolved,
            })
        else:
            final.append({"id": row_id, "source": a.get("source"),
                          "labels": resolved, "agreement": "tiebreak"})

    added = temporal_weak_supervision(rows_by_id, final)

    ensure_data_dir()
    write_jsonl(LABELS_FINAL, final)
    write_jsonl(REVIEW_QUEUE, needs_human)
    if needs_tiebreak:
        pending = [rows_by_id[i] for i in needs_tiebreak if i in rows_by_id]
        write_jsonl(LABELS_TIEBREAK.replace(".jsonl", "_input.jsonl"), pending)

    # 5% stratified audit sample — the user's second review duty, drawn from rows
    # the models AGREED on (agreement is the thing being audited: silent shared
    # bias is invisible to the disagreement queue by construction).
    rng = random.Random(SEED)
    agreed = [e for e in final if e["agreement"] == "both"]
    by_source = {}
    for entry in agreed:
        by_source.setdefault(entry.get("source", "?"), []).append(entry)
    audit = []
    for source_rows in by_source.values():
        rng.shuffle(source_rows)
        audit += source_rows[:max(1, int(len(source_rows) * AUDIT_FRACTION))]
    for entry in audit:
        row = rows_by_id.get(entry["id"], {})
        entry["text"] = row.get("text", "")
        entry["context_text"] = row.get("context_text")
    write_jsonl(AUDIT_SAMPLE, audit)

    if verbose:
        ctx_counts = Counter()
        for entry in final:
            for label in entry["labels"][CONTEXT_RELIANCE]:
                ctx_counts[label] += 1
            if not entry["labels"][CONTEXT_RELIANCE]:
                ctx_counts["(derived Zero_Shot)"] += 1
        intent_counts = Counter()
        for entry in final:
            for label in entry["labels"][INTENT]:
                intent_counts[label] += 1

        rate = len(final) / max(1, len(both))
        print(f"[merge] settled {len(final)}/{len(both)} ({rate:.1%})")
        print(f"[merge] per-head disagreement: {dict(head_disagreements)}")
        print(f"[merge] awaiting tiebreak: {len(needs_tiebreak)}")
        print(f"[merge] HUMAN REVIEW QUEUE: {len(needs_human)} → {REVIEW_QUEUE}")
        print(f"[merge] temporal detector added {added} Temporal_Recall positives")
        print(f"[merge] audit sample: {len(audit)} → {AUDIT_SAMPLE}")
        print(f"[merge] context labels: {dict(ctx_counts)}")
        print(f"[merge] intent labels: {dict(intent_counts)}")

        # The floors that decide whether a label is trainable at all (§4: a label
        # with <150 real positives gets dropped rather than trained as a coin flip).
        print("\n[merge] per-label positives vs the 300 floor:")
        for head_name in (INTENT, CONTEXT_RELIANCE):
            for label in schema.labels(head_name):
                n = sum(1 for e in final if label in e["labels"][head_name])
                flag = "OK " if n >= 300 else ("thin" if n >= 150 else "DROP?")
                print(f"    {flag} {label:24} {n}")
    return final, needs_human


def main():
    ap = argparse.ArgumentParser(description="B1 stage 4: two-labeler labeling + reconcile")
    ap.add_argument("--labeler", choices=list(LABELERS))
    ap.add_argument("--corpus", nargs="*",
                    default=[CORPUS_RAW, ICEDEV_STITCHED, CORPUS_SYNTH])
    ap.add_argument("--backend", default="vllm", choices=["sglang", "vllm"])
    ap.add_argument("--model", default=None, help="override the profile's model")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="dry run on N stratified rows")
    ap.add_argument("--out", default=None)
    ap.add_argument("--attach", action="store_true",
                    help="use an already-running server instead of launching one")
    ap.add_argument("--tiebreak-only", action="store_true",
                    help="label only the rows the merge flagged as disagreements")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge(args.corpus)
        return
    if not args.labeler:
        ap.error("pass --labeler A|B|C or --merge")

    profile = LABELERS[args.labeler]
    out_path = args.out or profile["out"]
    corpus = args.corpus
    if args.tiebreak_only:
        pending = LABELS_TIEBREAK.replace(".jsonl", "_input.jsonl")
        if not os.path.exists(pending):
            raise SystemExit(f"{pending} missing — run --merge first")
        corpus = [pending]

    rows = load_corpus(corpus, limit=args.limit)
    print(f"[label] labeler {args.labeler}: {len(rows)} corpus rows → {out_path}")

    server = LocalServer(args.model or profile["model"], backend=args.backend,
                         port=args.port, context_length=args.context_length,
                         max_running=args.concurrency)
    concurrency = args.concurrency or server.max_running

    if args.attach:
        if not is_up(server.base_url):
            raise SystemExit(f"nothing serving at {server.base_url}")
        model_id = served_model(server.base_url) or server.model
        asyncio.run(run_labeler(rows, out_path, server.base_url, model_id, concurrency))
        return

    with server:
        model_id = served_model(server.base_url) or server.model
        asyncio.run(run_labeler(rows, out_path, server.base_url, model_id, concurrency))


if __name__ == "__main__":
    main()
