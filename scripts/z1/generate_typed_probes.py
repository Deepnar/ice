#!/usr/bin/env python3
"""Z1: generate TYPED retrieval probes — one class per retrieval leg.

**Why this supersedes `generate_probes.py`.** That script generates exactly one
kind of probe: a question answerable from ONE turn. Measured consequence
(2026-08-13, 592 probes, production path): of 377 hits, the producing legs were
`bm25+vector` **376 times** and `vector` once. **Codex, procedural,
batch-summary and timeline scored zero — not rarely, never.** The gold is an
episodic row id and only episodic fragments carry one, so those legs cannot be
credited by construction (G48).

That is worse than an incomplete metric, because the unscoreable legs still
spend the token budget. Tuning leg weights against it would drive codex and
procedural toward zero and report an improvement — deleting the knowledge graph
and calling it tuning. **Typed probes are the fix, and they only work if the
SCORER BRANCHES ON THE TYPE**: recall@k is the wrong metric for four of the five
classes below, and generating prettier probes while scoring them all with
recall@k would change nothing.

| type | what it asks | how it must be scored |
|---|---|---|
| `episodic_lookup`   | one fact stated in one turn | recall@k on the gold turn |
| `codex_multihop`    | a fact assembled from 2+ turns via a shared entity | did the required ENTITIES appear (no single gold turn) |
| `procedural`        | a habit visible across a session | matched against the pattern, not a turn |
| `summary_synthesis` | something spanning many turns | coverage of the required turn SET |
| `temporal`          | what was true BEFORE a change | gold turn + an ordering constraint |

**The ambiguity guard applies to `episodic_lookup` ONLY.** For every other class,
several turns answering the question is the POINT, not a defect — applying the
guard there would reject exactly the probes that exercise the other legs.

Model: OpenCode Go (`PROBE_*` in .env). A browser `User-Agent` is mandatory —
without one Cloudflare answers 403/1010 and the endpoint looks broken.

Output: `experiments/curation_files/typed_probes.json` (gitignored — it quotes
personal conversation text).

Run:
  uv run python scripts/z1/generate_typed_probes.py --limit 3 --per-turn 1  # smoke
  uv run python scripts/z1/generate_typed_probes.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.z1.derive_retrieval_gt import (  # noqa: E402
    CURATION, build_idf, load_conversations, turn_text,
)
from scripts.z1.generate_probes import better_elsewhere, evidence_present  # noqa: E402
from scripts.z1.run_meta import file_digest, run_meta  # noqa: E402
from src.retrieval.ner_utils import extract_entities  # noqa: E402

_EMB = None


def _embedder():
    """The shared frozen encoder, loaded once — NER needs it for anchor terms."""
    global _EMB
    if _EMB is None:
        from src.memory.embedder import get_embedder
        _EMB = get_embedder()
    return _EMB

OUT = CURATION / "typed_probes.json"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/131.0.0.0 Safari/537.36")


def _env(name: str, default: str = "") -> str:
    if os.environ.get(name):
        return os.environ[name]
    envfile = Path(".env")
    if envfile.exists():
        for ln in envfile.read_text().splitlines():
            if ln.startswith(f"{name}=") and not ln.lstrip().startswith("#"):
                return ln.split("=", 1)[1].strip().strip('"').strip("'")
    return default


def call_model(prompt: str, *, schema: dict, max_tokens: int = 3000,
               retries: int = 3) -> dict:
    """One JSON-returning call to the probe model.

    ⚑ The User-Agent is not decoration. Cloudflare fronts this endpoint and
    answers **403, error 1010** to anything that looks like a scripted client —
    which reads as "the API is down" or "the key is wrong" and is neither.
    """
    key, base, model = (_env("PROBE_API_KEY"), _env("PROBE_API_BASE_URL"),
                        _env("PROBE_MODEL"))
    if not key or not base:
        raise SystemExit("PROBE_API_KEY / PROBE_API_BASE_URL missing from .env")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content":
             "You write evaluation probes for a memory-retrieval system. You "
             "return ONLY valid JSON matching the requested shape. No prose, no "
             "code fences."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{base.rstrip('/')}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}",
                         "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except Exception as exc:                       # noqa: BLE001
            # Only TRANSPORT failures are worth retrying.
            last = exc
            time.sleep(2 * (attempt + 1))
            continue

        txt = txt.strip()
        # ⚑ An EMPTY reply is a legitimate answer, not an error. Several prompts
        # end with "if there is nothing here, return an empty list", and the
        # model obliges by returning nothing at all — whereupon `json.loads("")`
        # raises at char 0. The first version treated that as a failure and
        # retried it three times with backoff, so every turn that honestly had
        # no probe cost ~12 seconds and printed an alarming error. Measured on
        # the first smoke run: 8 of 14 calls.
        if not txt:
            return {"probes": [], "_empty": True}
        if txt.startswith("```"):
            parts = txt.split("```")
            txt = parts[1] if len(parts) > 1 else txt
            txt = txt[4:] if txt.lower().startswith("json") else txt
            txt = txt.strip()
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # Prose around the JSON — salvage the outermost object rather than
            # throwing away a good generation over a stray sentence.
            i, j = txt.find("{"), txt.rfind("}")
            if i != -1 and j > i:
                try:
                    return json.loads(txt[i:j + 1])
                except json.JSONDecodeError:
                    pass
            # ⚑ TRUNCATED REPLY: salvage the probes that ARE complete instead of
            # discarding the whole generation. A reply cut off mid-object still
            # contains N valid ones before the cut, and throwing them away is
            # how the codex class — the one this whole probe set exists for —
            # came back with 3 probes from 147 prompts.
            salvaged, depth, start = [], 0, None
            for idx, ch in enumerate(txt):
                if ch == "{":
                    if depth == 0:
                        start = idx
                    depth += 1
                elif ch == "}" and depth:
                    depth -= 1
                    if depth == 0 and start is not None:
                        frag = txt[start:idx + 1]
                        try:
                            obj = json.loads(frag)
                            if isinstance(obj, dict) and obj.get("question"):
                                salvaged.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start = None
            if salvaged:
                print(f"    ~ salvaged {len(salvaged)} probe(s) from a truncated reply")
                return {"probes": salvaged}
            print(f"    ! unparsable reply ({len(txt)} chars): {txt[:100]!r}")
            return {"probes": []}
    print(f"    ! transport failure after {retries}: {type(last).__name__}: {last}")
    return {"probes": []}


_SHAPE = ('{"probes":[{"question":"...","answer":"...","evidence":"...",'
          '"why_this_type":"..."}]}')

_progress = {"done": 0, "total": 0}
_plock = threading.Lock()


def run_tasks(tasks: list, workers: int) -> list:
    """Run every prompt concurrently and return results in the SAME order.

    The generator is entirely API-latency-bound — a 2-turn smoke slice took
    3m19s wall-clock with essentially zero CPU, and a full corpus is ~395 calls.
    Sequentially that is hours of waiting on a socket, which makes the generator
    something you run once and never iterate on. Order is preserved because each
    result has to be matched back to the turn it came from.
    """
    _progress["done"], _progress["total"] = 0, len(tasks)

    def one(task):
        result = call_model(task["prompt"], schema={})
        with _plock:
            _progress["done"] += 1
            d, t = _progress["done"], _progress["total"]
            if d % 10 == 0 or d == t:
                print(f"    {d}/{t} calls", flush=True)
        return result

    # ⚑ CHECKPOINT EVERY CALL. This function is the expensive part — a full
    # run is HOURS of GPU — and the caller previously wrote nothing until the
    # very end, so a run killed at 99% produced exactly what a run killed at 1%
    # produced. That happened: a two-hour generation was lost entirely. Raw
    # replies are appended to a sidecar as they land, so a killed run can be
    # resumed or salvaged instead of repeated.
    ckpt = OUT.with_suffix(".raw-calls.jsonl")

    def _checkpoint(task, result):
        try:
            with ckpt.open("a") as fh:
                fh.write(json.dumps({"kind": task.get("kind"),
                                     "conversation": task.get("conversation"),
                                     "result": result}, default=str) + "\n")
        except Exception:                                        # noqa: BLE001
            pass                                                 # never fail the run over telemetry

    def one_ckpt(t):
        r = one(t)
        _checkpoint(t, r)
        return r

    if workers <= 1:
        return [one_ckpt(t) for t in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one_ckpt, tasks))


def prompt_episodic(turn: dict, n: int) -> str:
    return (
        "Below is ONE turn from a real conversation between a person and an AI.\n\n"
        f"--- TURN ---\n{turn_text(turn)[:5000]}\n--- END TURN ---\n\n"
        f"Write {n} questions this person might ask LATER, each answerable from "
        "THIS turn alone.\n\n"
        "TYPE: episodic_lookup — a single stated fact in a single turn.\n\n"
        "Rules:\n"
        "1. Write the way THIS person types: their register, their informality, "
        "lowercase if that is how they write.\n"
        "2. Never refer to the turn itself ('according to the above', 'as "
        "mentioned'). They are asking from memory, having forgotten details.\n"
        "3. Ask about something CONCRETE and stated: a fact, a name, a number, a "
        "decision, a stated reason.\n"
        "4. ⚑ The question must contain enough of its own specifics to be "
        "findable WITHOUT the answer. 'what did we decide?' is useless — a dozen "
        "turns answer it. Name the thing being asked about.\n"
        "5. `evidence` is a VERBATIM span from the turn containing the answer.\n"
        "6. `answer` is short and direct.\n"
        "7. `why_this_type` states in a few words which single fact it targets.\n\n"
        f"Return JSON: {_SHAPE}\n"
        "If the turn holds nothing concrete, return an empty probes list."
    )


def prompt_codex(turns: list, entity: str, n: int) -> str:
    joined = "\n\n".join(f"[turn {t.get('turn_number')}]\n{turn_text(t)[:1800]}"
                         for t in turns)
    return (
        f"Below are SEVERAL turns that all mention **{entity}**.\n\n"
        f"--- TURNS ---\n{joined}\n--- END TURNS ---\n\n"
        f"Write {n} questions that CANNOT be answered from any single turn above "
        "— each must require joining facts from at least TWO of them.\n\n"
        "TYPE: codex_multihop — this tests the knowledge graph, which stores "
        "entities and the relations between them.\n\n"
        "Rules:\n"
        "1. The question must name a real entity from the turns "
        f"(such as {entity}) so it can be looked up.\n"
        "2. Answering it must need a CONNECTION — two facts about the same "
        "entity, or a chain from one entity to another. Example shape: 'what did "
        "I end up using X for, given why I picked it?'\n"
        "3. It must be a natural thing this person would ask, not a quiz.\n"
        "4. `evidence` is ONLY the turn numbers you used, e.g. \"1, 4\". Do NOT "
        "quote the turns back — it makes the reply long and adds nothing.\n"
        "5. `why_this_type` names the entities and the hop, e.g. "
        "'needs X's purpose from turn 4 AND X's cost from turn 9'.\n\n"
        f"Return JSON: {_SHAPE}\n"
        "If no genuine multi-turn connection exists, return an empty list. "
        "Do NOT pad with single-turn questions — a wrong type is worse than none."
    )


def prompt_procedural(user_prompts: list, n: int) -> str:
    numbered = "\n".join(f"{i}. {p[:400]}" for i, p in enumerate(user_prompts, 1))
    return (
        f"Below are {len(user_prompts)} messages the SAME person wrote during one "
        "session, in order.\n\n"
        f"{numbered}\n\n"
        f"Write {n} questions this person might later ask ABOUT THEIR OWN "
        "HABITS — patterns visible across these messages.\n\n"
        "TYPE: procedural — this tests stored behavioural patterns, not events.\n\n"
        "Rules:\n"
        "1. The question must be about a RECURRING way they behave, not a single "
        "thing that happened. 'do i always ask for the plan first?' is right; "
        "'what did i say about the plan?' is the wrong type.\n"
        "2. The habit must be visible in at least TWO of the messages.\n"
        "3. `evidence` cites the message numbers that show it, e.g. '2, 5, 6'.\n"
        "4. `answer` states the habit in one sentence.\n"
        "5. `why_this_type` says which behaviour repeats.\n\n"
        f"Return JSON: {_SHAPE}\n"
        "If nothing genuinely repeats, return an empty list."
    )


def prompt_temporal(before: dict, after: dict, n: int) -> str:
    return (
        "Below are two turns from the same conversation, in order. Something "
        "CHANGED between them.\n\n"
        f"--- EARLIER (turn {before.get('turn_number')}) ---\n"
        f"{turn_text(before)[:2200]}\n\n"
        f"--- LATER (turn {after.get('turn_number')}) ---\n"
        f"{turn_text(after)[:2200]}\n--- END ---\n\n"
        f"Write {n} questions about the EARLIER state — what was true before the "
        "change.\n\n"
        "TYPE: temporal — this tests whether memory can return a superseded fact "
        "when explicitly asked for the past, instead of only the current one.\n\n"
        "Rules:\n"
        "1. The question must explicitly ask about the past: 'what was i using "
        "BEFORE i switched', 'what did i originally plan'.\n"
        "2. It must have a different answer now than it did then — otherwise it "
        "is not this type.\n"
        "3. `answer` is the EARLIER value.\n"
        "4. `evidence` quotes the earlier turn verbatim.\n"
        "5. `why_this_type` names what changed and in which direction.\n\n"
        f"Return JSON: {_SHAPE}\n"
        "If nothing actually changed between these turns, return an empty list."
    )


def prompt_synthesis(turns: list, n: int) -> str:
    joined = "\n\n".join(f"[turn {t.get('turn_number')}]\n{turn_text(t)[:900]}"
                         for t in turns)
    return (
        f"Below is a span of {len(turns)} consecutive turns from one "
        f"conversation.\n\n--- SPAN ---\n{joined}\n--- END SPAN ---\n\n"
        f"Write {n} questions that need MOST of this span to answer properly — "
        "the kind answered by a summary rather than by one turn.\n\n"
        "TYPE: summary_synthesis — this tests batch summaries and clustering.\n\n"
        "Rules:\n"
        "1. The question must be genuinely broad: 'how did my thinking about X "
        "change over all this', 'what were all the options i weighed'.\n"
        "2. It must NOT be answerable from any single turn — if one turn covers "
        "it, it is the wrong type.\n"
        "3. `answer` is a short synthesis.\n"
        "4. `evidence` lists the turn numbers required, e.g. '12, 14, 17, 19'.\n"
        "5. `why_this_type` says why one turn cannot answer it.\n\n"
        f"Return JSON: {_SHAPE}\n"
        "If the span has no coherent thread, return an empty list."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-turn", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N turns per conversation (smoke)")
    ap.add_argument("--only", default=None, help="one conversation slug")
    ap.add_argument("--types", default="all",
                    help="comma list: episodic,codex,procedural,temporal,synthesis")
    ap.add_argument("--min-evidence-overlap", type=float, default=0.6)
    ap.add_argument("--codex-anchors", type=int, default=12,
                    help="entities per conversation to build multi-hop probes around")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent API calls; the run is latency-bound")
    ap.add_argument("--session-size", type=int, default=14,
                    help="turns per procedural/synthesis block")
    args = ap.parse_args()

    want = ({"episodic", "codex", "procedural", "temporal", "synthesis"}
            if args.types == "all" else set(args.types.split(",")))

    convs = load_conversations()
    if args.only:
        convs = {k: v for k, v in convs.items() if k.startswith(args.only)}
    if not convs:
        print("no conversations loaded")
        return 1

    probes, stats = [], Counter()

    # ── Phase 1: build every prompt, run none of them ───────────────────────
    # Split from execution so the whole corpus can be issued concurrently; the
    # generator is latency-bound, not CPU-bound.
    tasks = []
    for slug, rec in convs.items():
        turns = rec["turns"]
        if args.limit:
            turns = turns[:args.limit]
        conv8 = slug[:8]
        ctx = {"slug": slug, "conv8": conv8, "turns": turns}

        if "episodic" in want:
            for t in turns:
                tasks.append({"kind": "episodic", "prompt": prompt_episodic(t, args.per_turn),
                              "turn": t, **ctx})
        if "codex" in want:
            # ⚑ ANCHORS COME FROM ICE'S OWN NER, not word frequency.
            # The first version ranked words longer than 4 characters by
            # document frequency, which on real conversation produces
            # `story`, `something`, `because`, `that's`, `everything` — and then
            # asked the model for a multi-hop question about "because". It
            # correctly returned nothing, and codex probes came out at ZERO.
            # Using the same NER the codex extractor uses means every anchor is
            # a term the graph plausibly holds a node for, which is the whole
            # point of this probe class.
            ents = Counter()
            for t in turns:
                try:
                    for e in extract_entities(turn_text(t)[:3000], _embedder()) or []:
                        e = e.strip().lower()
                        if len(e) > 2:
                            ents[e] += 1
                except Exception as exc:                      # noqa: BLE001
                    print(f"    ! NER failed on a turn: {type(exc).__name__}: {exc}")
            for entity, _ in ents.most_common(args.codex_anchors):
                hosts = [t for t in turns
                         if entity in turn_text(t).lower()][:4]
                if len(hosts) >= 2:
                    tasks.append({"kind": "codex", "entity": entity, "hosts": hosts,
                                  "prompt": prompt_codex(hosts, entity, args.per_turn),
                                  **ctx})
        blocks = [turns[i:i + args.session_size]
                  for i in range(0, len(turns), args.session_size)]
        for blk in blocks:
            if len(blk) < 4:
                continue
            if "procedural" in want:
                ups = [(t.get("user_input") or "").strip() for t in blk]
                ups = [u for u in ups if u]
                if len(ups) >= 4:
                    tasks.append({"kind": "procedural", "block": blk,
                                  "prompt": prompt_procedural(ups, args.per_turn), **ctx})
            if "synthesis" in want:
                tasks.append({"kind": "synthesis", "block": blk,
                              "prompt": prompt_synthesis(blk, args.per_turn), **ctx})
        if "temporal" in want and len(turns) >= 8:
            step = max(1, len(turns) // 8)
            for i in range(0, len(turns) - step, step * 2):
                a = turns[i]
                b = turns[min(i + step * 2, len(turns) - 1)]
                tasks.append({"kind": "temporal", "before": a, "after": b,
                              "prompt": prompt_temporal(a, b, 1), **ctx})

    print(f"built {len(tasks)} prompts across {len(convs)} conversations; "
          f"running {args.workers} at a time")
    t0 = time.time()
    results = run_tasks(tasks, args.workers)
    print(f"  all calls done in {time.time() - t0:.0f}s")

    # ── Phase 2: assemble ───────────────────────────────────────────────────
    idf_cache = {}
    for task, got in zip(tasks, results):
        conv8, turns, kind = task["conv8"], task["turns"], task["kind"]
        if kind == "episodic":
            if task["slug"] not in idf_cache:
                idf_cache[task["slug"]] = build_idf(turns)
            idf = idf_cache[task["slug"]]
            t = task["turn"]
            tn = t.get("turn_number")
            for p in (got.get("probes") or []):
                q, ev = (p.get("question") or "").strip(), (p.get("evidence") or "")
                if not q:
                    continue
                if not evidence_present(ev, t, args.min_evidence_overlap):
                    stats["episodic_evidence_missing"] += 1
                    continue
                # Guard applies to THIS TYPE ONLY (see module docstring).
                if better_elsewhere(q, p.get("answer", ""), tn, turns, idf) is not None:
                    stats["episodic_ambiguous"] += 1
                    continue
                probes.append({
                    "probe_type": "episodic_lookup", "conversation": conv8,
                    "gold_turns": [tn], "question": q,
                    "answer": p.get("answer", ""), "evidence": ev,
                    "why_this_type": p.get("why_this_type", ""),
                    "scoring": "recall_at_k",
                })
                stats["episodic_kept"] += 1

        elif kind == "codex":
            hosts = task["hosts"]
            for p in (got.get("probes") or []):
                q = (p.get("question") or "").strip()
                if not q:
                    continue
                probes.append({
                    "probe_type": "codex_multihop", "conversation": conv8,
                    "gold_turns": [h.get("turn_number") for h in hosts],
                    "anchor_entity": task["entity"], "question": q,
                    "answer": p.get("answer", ""),
                    "evidence": p.get("evidence", ""),
                    "why_this_type": p.get("why_this_type", ""),
                    "scoring": "entity_coverage",
                })
                stats["codex_kept"] += 1

        elif kind == "procedural":
            blk = task["block"]
            for p in (got.get("probes") or []):
                q = (p.get("question") or "").strip()
                if not q:
                    continue
                probes.append({
                    "probe_type": "procedural", "conversation": conv8,
                    "gold_turns": [b.get("turn_number") for b in blk],
                    "question": q, "answer": p.get("answer", ""),
                    "evidence": p.get("evidence", ""),
                    "why_this_type": p.get("why_this_type", ""),
                    "scoring": "pattern_match",
                })
                stats["procedural_kept"] += 1

        elif kind == "synthesis":
            blk = task["block"]
            for p in (got.get("probes") or []):
                q = (p.get("question") or "").strip()
                if not q:
                    continue
                probes.append({
                    "probe_type": "summary_synthesis", "conversation": conv8,
                    "gold_turns": [b.get("turn_number") for b in blk],
                    "question": q, "answer": p.get("answer", ""),
                    "evidence": p.get("evidence", ""),
                    "why_this_type": p.get("why_this_type", ""),
                    "scoring": "turn_set_coverage",
                })
                stats["synthesis_kept"] += 1

        elif kind == "temporal":
            a, b = task["before"], task["after"]
            for p in (got.get("probes") or []):
                q = (p.get("question") or "").strip()
                if not q:
                    continue
                probes.append({
                    "probe_type": "temporal", "conversation": conv8,
                    "gold_turns": [a.get("turn_number")],
                    "superseded_by": b.get("turn_number"), "question": q,
                    "answer": p.get("answer", ""),
                    "evidence": p.get("evidence", ""),
                    "why_this_type": p.get("why_this_type", ""),
                    "scoring": "recall_at_k_with_order",
                })
                stats["temporal_kept"] += 1

    # ⚑ PROVENANCE FIRST. Every experiment artifact in this repo carries a
    # run_meta block — model, commit, dirty state, resolved settings, corpus
    # digests, per-type counts and the rejection tallies. A probe set without
    # it is a pile of questions nobody can date, attribute or reproduce, and
    # this cycle produced several numbers that had to be re-derived for exactly
    # that reason.
    counts = dict(Counter(p["probe_type"] for p in probes))
    meta = run_meta(
        script=__file__, args=vars(args),
        settings_keys=["probe_model", "probe_api_base_url", "probe_api_key"],
        inputs=[file_digest(f) for f in sorted(CURATION.glob("EC-*.json"))],
        extra={
            "model": _env("PROBE_MODEL"),
            "conversations": {c[:8]: len(r["turns"]) for c, r in convs.items()},
            "prompts_issued": len(tasks),
            "probes_by_type": counts,
            "rejections": dict(stats),
            "wall_clock_seconds": round(time.time() - t0),
            "ambiguity_guard": "question-only (G46); applied to episodic_lookup ONLY",
            "known_limits": [
                "three conversations only — more probes is not more independent evidence",
                "four of five types are NOT scoreable by recall@k; see `scoring` per probe",
                "procedural/synthesis `evidence` cites message numbers within the "
                "block, not absolute turn numbers",
            ],
        })
    # ⚑ THIS FILE IS AN ASSET, NOT AN OUTPUT. It is expensive to build, it is
    # gitignored so there is no history to recover from, and every scoring run
    # depends on it. On 2026-08-16 a `--types temporal --limit 6` smoke test
    # built 0 prompts and wrote the file anyway, destroying 420 probes — 48
    # codex probes were unrecoverable, and those had cost a salvage fix to get
    # from 12 to 89 in the first place. Three guards, cheapest first.
    existing = []
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text()).get("probes", [])
        except Exception:                                        # noqa: BLE001
            existing = []

    # 1. Never let an empty or failed run overwrite real probes.
    if not probes and existing:
        print(f"\n⚠ REFUSING TO WRITE: generated 0 probes and {OUT} holds "
              f"{len(existing)}. Nothing written; the existing set is intact.")
        return 1

    # 2. A partial run (--types / --limit) must not silently drop the types it
    #    did not generate. Merge by question text, new wins on collision.
    if existing:
        merged = {p["question"]: p for p in existing}
        added = sum(1 for p in probes if p["question"] not in merged)
        merged.update({p["question"]: p for p in probes})
        if len(merged) > len(probes):
            print(f"\n  merging into the existing set: {len(existing)} on disk "
                  f"+ {added} new -> {len(merged)}")
        probes = list(merged.values())
        counts = Counter(p["probe_type"] for p in probes)

    # 3. Keep the previous generation recoverable regardless.
    if existing:
        bak = OUT.with_suffix(".prev.json")
        bak.write_text(json.dumps({"probes": existing}, indent=1))

    OUT.write_text(json.dumps({
        "meta": meta,
        "model": _env("PROBE_MODEL"), "generated_utc": time.strftime("%Y%m%dT%H%M%SZ"),
        "counts": counts,
        "probes": probes,
    }, indent=1))

    print(f"\n{'='*58}\nTYPED PROBES\n{'='*58}")
    for k, v in sorted(Counter(p["probe_type"] for p in probes).items()):
        print(f"  {k:<20} {v}")
    print(f"  {'TOTAL':<20} {len(probes)}")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}")
    print(f"\nwrote {OUT}")
    print("\n⚠ These are NOT scoreable by recall@k alone. Four of the five types "
          "carry a different `scoring` field and the scorer must branch on it "
          "(G48) — otherwise this changes nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
