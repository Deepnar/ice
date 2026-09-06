#!/usr/bin/env python3
"""LongMemEval on ICE v2 (tag `v2-paper-eval`) -- resumable, unattended runner.

⚑ RUN THIS FROM THE WORKTREE so `uv` resolves v2's lockfile and `src.*` imports
resolve to the frozen tree. v2 embeds at truncate_dim=384; `main` embeds at 1024.
Running under main's venv produces silently wrong vectors.

    cd /home/deepnar/Programs/ice-worktrees/v2-paper-eval
    uv run python /home/deepnar/Programs/ice/experiments/lme/lme_run.py --phase oracle

RESUMABILITY -- the design constraint. One answer file per instance, written
atomically (tmp + os.replace). **Presence of the file IS the state**; there is no
progress file to fall out of sync with reality. Kill this at any time -- Ctrl-C,
lid close, power cut. The next invocation skips finished instances and redoes any
interrupted one from a full store wipe, which is what correctness demands anyway.

WHAT THIS DOES NOT DO: simulate decay days. LSREP ages memory between checkpoints
because it is measuring accumulation. LongMemEval asks once, at the end, so there
is nothing to age between; simulating decay here would model something the corpus
does not contain. Clustering runs once after ingestion so cluster-scoped retrieval
has something to scope to. Recorded as a deliberate deviation, not an oversight.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# This file lives on `main` but must import `src.*` from the WORKTREE we are run
# from. Python seeds sys.path with the script's directory, not the cwd, so the
# worktree root goes on explicitly -- the same thing tests/ does.
sys.path.insert(0, os.getcwd())

# --- repo layout -------------------------------------------------------------
HARNESS_DIR = Path(__file__).resolve().parent
DATA_DIR = HARNESS_DIR / "data"
DEFAULT_OUT = HARNESS_DIR / "runs" / "v2-paper-eval"
ADAPTER_VERSION = "ice-v2-lme-sessions-v2"

_stop_requested = False
_signal_count = 0
_active_stage = "starting"


def _on_signal(signum, _frame):
    """Stop gracefully once; let a second signal abort the active instance."""
    global _signal_count, _stop_requested
    _signal_count += 1
    if _signal_count > 1:
        print(
            f"\n  [signal {signum}] aborting {_active_stage} now. "
            "Completed answers and ingestion checkpoints remain resumable.",
            flush=True,
        )
        raise SystemExit(128 + signum)
    _stop_requested = True
    print(
        f"\n  [signal {signum}] graceful stop requested during {_active_stage}; "
        "finishing the current instance, then stopping. Press Ctrl-C again to "
        "abort it now. Re-run the same command to resume.",
        flush=True,
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write via tmp + os.replace. POSIX rename is atomic: a reader never sees
    a half-written answer, and a crash mid-write leaves the old file or none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def _progress(done: int, total: int, prefix: str, started: float) -> None:
    """Live bar on STDERR, so it stays a TTY even though stdout is piped to tee.

    Writing it to stdout would fill run.log with thousands of carriage-returned
    redraws; writing it to stderr keeps the log readable and the screen live.
    """
    if not sys.stderr.isatty():
        return
    pct = 100 * done / total if total else 0
    el = time.time() - started
    eta = (el / done) * (total - done) if done else 0
    sys.stderr.write(f"\r  {prefix} {_bar(done, total)} {done}/{total} "
                     f"({pct:5.1f}%)  {_fmt(el)} elapsed  eta {_fmt(eta)}   ")
    sys.stderr.flush()


def _progress_done() -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\r" + " " * 110 + "\r")
        sys.stderr.flush()


def _fmt(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# --- phases ------------------------------------------------------------------
# Sized against the measured corpus: median 492 turns/instance. See README.
PHASES = {
    # Control. Evidence sessions only -- ingestion is a rounding error. Proves the
    # adapter end-to-end before anything expensive. If ICE cannot answer from the
    # evidence sessions alone, the defect is HERE, not in ICE's memory.
    "oracle": {"corpus": "longmemeval_oracle", "select": "all"},
    # All 30 `_abs` instances. A complete, non-arbitrary unit, and the sharpest
    # test of a curation-first system: decline, or confabulate from retrieved noise?
    "abstention": {"corpus": "longmemeval_s", "select": "abstention"},
    # Stratified widening. Only after oracle and abstention look sane.
    "stratified": {"corpus": "longmemeval_s", "select": "stratified"},
    # Complete public LongMemEval-S.  This is deliberately a separate phase from
    # the old stratified sample so its artifact root cannot be misread as a full
    # benchmark run.
    "full": {"corpus": "longmemeval_s", "select": "all"},
}


def select_instances(corpus: list[dict], how: str, limit: int | None, seed: int) -> list[dict]:
    import random
    if how == "abstention":
        chosen = [x for x in corpus if x["question_id"].endswith("_abs")]
    elif how == "stratified":
        by_type: dict[str, list] = {}
        for x in corpus:
            by_type.setdefault(x["question_type"], []).append(x)
        rng = random.Random(seed)
        per = max(1, (limit or len(corpus)) // max(1, len(by_type)))
        chosen = []
        for _t, xs in sorted(by_type.items()):
            chosen.extend(rng.sample(xs, min(per, len(xs))))
    else:
        chosen = list(corpus)
    chosen.sort(key=lambda x: x["question_id"])  # deterministic order
    return chosen[:limit] if limit else chosen


# --- environment capture -----------------------------------------------------
def capture_env(db_url: str) -> dict:
    """Record what actually ran. PROVENANCE.md's standing rule: a run that cannot
    say what produced it is not a result."""
    def _sh(cmd, cwd=None):
        try:
            return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                  timeout=15).stdout.strip()
        except Exception:
            return "unavailable"

    env = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "system_under_test": "ICE v2 @ tag v2-paper-eval",
        "worktree_head": _sh(["git", "rev-parse", "HEAD"], cwd=os.getcwd()),
        "worktree_describe": _sh(["git", "describe", "--tags", "--always"], cwd=os.getcwd()),
        "harness_head": _sh(["git", "rev-parse", "HEAD"], cwd=str(HARNESS_DIR)),
        "database_url": db_url.rsplit("@", 1)[-1],  # host/db only, never credentials
    }
    for mod in ("torch", "transformers", "sentence_transformers", "pgvector"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "unavailable"

    manifest = DATA_DIR / "MANIFEST.json"
    if manifest.exists():
        env["corpus"] = json.loads(manifest.read_text())
    return env


# --- store lifecycle ---------------------------------------------------------
def wipe_store(SessionLocal, Base) -> None:
    """Every LongMemEval instance needs its OWN memory state -- haystacks cannot
    be shared. Truncating all mapped tables is derived from ORM metadata rather
    than a hand-listed set, so a schema change cannot leave a table un-wiped and
    silently leak evidence between instances."""
    from sqlalchemy import text
    db = SessionLocal()
    try:
        names = [t.name for t in Base.metadata.sorted_tables if t.name != "alembic_version"]
        if not names:
            raise RuntimeError("no mapped tables found -- refusing to run with an unknown schema")
        db.execute(text(f"TRUNCATE TABLE {', '.join(names)} RESTART IDENTITY CASCADE"))
        db.commit()
    finally:
        db.close()


def ingest_instance(layout, classifier, embedder, SessionLocal, models,
                    progress_prefix: str = "") -> int:
    """Replay one haystack, mirroring the sequence that produced the paper's
    numbers (experiments/mature/run_mature_experiment.py).

    ⚑ Assistant text is stored VERBATIM. LongMemEval's fixed assistant replies
    carry evidence -- `single-session-assistant` is 56 instances -- so letting ICE
    regenerate them would destroy the haystack.
    """
    from src.workers.post_flight import is_lossless, generate_summary
    from src.workers.codex_extractor import extract_triplets, handle_triplet
    EpisodicMemory = models["EpisodicMemory"]

    stored = 0
    total_pairs = sum(len(spec["pairs"]) for spec in layout["sessions"])
    ing_started = time.time()
    db = SessionLocal()
    try:
        for spec in layout["sessions"]:
            sid = spec["session_id"]
            cid = spec["conversation_id"]
            base_ts = spec["timestamp"]

            # Sessions carry ONE date but many turns; nudge within the session so
            # ordering is preserved and the deterministic batch_id stays unique.
            pairs = spec["pairs"]
            for t_idx, (prompt, response) in enumerate(pairs):
                ts = base_ts + timedelta(seconds=t_idx)
                batch_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{cid}:{sid}:{t_idx}")
                if db.query(EpisodicMemory).filter_by(batch_id=batch_id).first():
                    continue  # idempotent: a partial replay resumes cleanly

                classification = classifier.classify(prompt)
                emb = embedder.encode(prompt, convert_to_tensor=False).tolist()
                turn = EpisodicMemory(
                    conversation_id=cid,
                    batch_id=batch_id,
                    timestamp=ts,
                    topic_tags=classification.topic_tags,
                    intent_tags=classification.intent_tags,
                    context_reliance=classification.context_reliance,
                    raw_text=f"User: {prompt}\n\nAssistant: {response}",
                    embedding=emb,
                    idempotency_key=str(uuid.uuid4()),
                )
                db.add(turn)
                db.flush()

                lossless = is_lossless(response)
                force_lossless = bool(
                    turn.topic_tags and ("Creative_&_Media" in turn.topic_tags
                                         or "Emotional_Processing" in (turn.intent_tags or []))
                )
                if force_lossless:
                    lossless, inject_raw, summary = True, True, None
                else:
                    full_text = f"User: {prompt}\nAssistant: {response}"
                    word_count = len(full_text.split())
                    has_code = "```" in response
                    inject_raw = True
                    if lossless and word_count > 500 and not has_code:
                        summary, inject_raw = generate_summary(prompt, response), False
                    elif not lossless:
                        summary, inject_raw = generate_summary(prompt, response), False
                    else:
                        summary = None
                turn.lossless_flag = lossless
                turn.summary_text = summary
                turn.inject_raw = inject_raw

                if lossless:
                    for triplet in extract_triplets(turn.raw_text, topic_tags=turn.topic_tags):
                        if not isinstance(triplet, dict):
                            continue
                        s, r, o = (triplet.get("subject"), triplet.get("relation"),
                                   triplet.get("object"))
                        if all(isinstance(v, str) and v.strip() for v in (s, r, o)):
                            handle_triplet(db, s.strip(), r.strip(), o.strip(), str(batch_id))
                db.commit()
                stored += 1
                _progress(stored, total_pairs, progress_prefix, ing_started)
                if stored % 5 == 0:    # durable line for the log file (the live
                                       # bar goes to stderr and is not logged)
                    print(f"           ... {stored}/{total_pairs} pairs "
                          f"({_fmt(time.time() - ing_started)})", flush=True)
                if _stop_requested:
                    # Stop between turns, never mid-write. The instance has no
                    # answer file yet, so the next run redoes it from a clean wipe.
                    break
            if _stop_requested:
                break
    finally:
        db.close()
        _progress_done()
    return stored


# --- answering -----------------------------------------------------------------
# Pinned to the mature harness so an LME-v2 number is comparable to the paper's
# own numbers rather than to a differently-answered variant of them.
# experiments/mature/run_mature_experiment.py:48-49, 636-637.
SINGLE_MODEL = "gemma4:26b-a4b-it-q4_K_M"
OLLAMA_URL = "http://localhost:11434/v1"
ANSWER_TEMPERATURE = 0.7
ANSWER_MAX_TOKENS = 20000
CONDITIONS = ("full_ice", "vector_rag")
EXPECTED_BACKGROUND_MODEL = "qwen3:4b-instruct-bg"


def answer_identity_matches(answer: dict | None, profile) -> bool:
    if not isinstance(answer, dict) or answer.get("model") != profile.model:
        return False
    recorded = answer.get("provider_endpoint")
    if recorded is None:
        return profile.name == "local-gemma26"
    return (recorded == profile.endpoint
            and answer.get("provider_profile") == profile.name)


def answer_complete_for_profile(answer: dict | None, profile) -> bool:
    return (
        answer_identity_matches(answer, profile)
        and answer.get("status") != "mute"
        and bool((answer.get("answer") or "").strip())
    )


def estimate_tokens(text: str) -> int:
    """The paper's own estimator (run_mature_experiment.py:226). Kept identical so
    the budget setter sees the same numbers it saw during the paper run."""
    return int(len(text.split()) * 1.33)


def answer_instance(inst, query_cid, condition, classifier, embedder,
                    SessionLocal, generator, models,
                    answer_max_tokens: int = ANSWER_MAX_TOKENS,
                    provider_session_id: str | None = None) -> dict:
    """Answer one question under one condition.

    ⚑ Mirrors v2 production auto memory: classify in a new conversation, derive
    the budget from that current conversation, retrieve globally with empty
    scope, then assemble against the same new conversation.
    """
    from sqlalchemy import bindparam, text as sql_text
    from pgvector.sqlalchemy import Vector as PgVector
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    from src.api.prompt_assembler import assemble_prompt
    MemorySlot = models["MemorySlot"]

    question = inst["question"]
    query_cid_str, auto_scope = auto_query_context(query_cid)
    classification = classifier.classify(
        question, conversation_id=query_cid_str
    )
    emb = embedder.encode(question, convert_to_tensor=False).tolist()

    db = SessionLocal()
    try:
        orchestrator = HybridRetrievalOrchestrator(db, embedder)
        # The question is asked in a NEW auto-scoped chat. v2 production derives
        # its budget from the current chat, not from every historical auto chat.
        # Passing the whole haystack length here fabricated a ~14k budget that a
        # real cross-session request never receives.
        orchestrator.set_budget_from_turn_count(0, 0,
                                                classification=classification)
        if condition == "vector_rag":
            # The paper's baseline verbatim: single-leg pgvector, top-30, no decay
            # weighting, no fusion, no classification-driven routing, no budget.
            # Auto memory is global, so search every history-session conversation.
            query = sql_text("""
                SELECT raw_text, summary_text, lossless_flag, inject_raw,
                    1 - (embedding <=> :prompt_embedding) as score
                FROM episodic_memory
                WHERE embedding IS NOT NULL AND is_archived = false
                ORDER BY score DESC LIMIT 30
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = db.execute(query, {"prompt_embedding": emb}).fetchall()
            fragments = [
                r.raw_text if r.lossless_flag else (r.summary_text or r.raw_text[:300])
                for r in rows
            ]
            context = "\n\n".join(fragments)
            messages = [{"role": "user",
                         "content": f"Context:\n{context}\n\nQuestion: {question}"}]
            n_frags = len(fragments)
        else:
            # v2 production leaves scope empty for memory_scope_type="auto";
            # retrieval searches globally across the user's past conversations.
            scope = auto_scope
            retrieved = orchestrator.retrieve(
                classification=classification,
                conversation_id=query_cid_str,
                prompt_embedding=emb,
                scope=scope,
            )
            memory_slots = db.query(MemorySlot).filter_by(is_active=True).all()
            messages = assemble_prompt(
                memory_slots=memory_slots,
                retrieved_fragments=retrieved,
                user_message=question,
                db_session=db,
                conversation_id=query_cid,
                classification=classification,
                scope=scope,
                max_recent_tokens=orchestrator.recent_token_budget,
            )
            n_frags = len(retrieved)

        injected = sum(estimate_tokens(m.get("content", "")) for m in messages)
        t0 = time.time()
        generation = generator.generate(
            messages,
            temperature=ANSWER_TEMPERATURE,
            max_output_tokens=answer_max_tokens,
            session_id=provider_session_id,
        )
        return {
            "answer": generation.text,
            "status": "complete" if generation.text.strip() else "mute",
            "model": generator.profile.model,
            "provider_profile": generator.profile.name,
            "provider_endpoint": generator.profile.endpoint,
            "provider_response_id": generation.response_id,
            "provider_usage": generation.usage,
            "provider_session_id": provider_session_id,
            "fragments": n_frags,
            "tokens_injected": injected,
            "seconds": round(time.time() - t0, 1),
            "topic_tags": list(classification.topic_tags or []),
            "intent_tags": list(classification.intent_tags or []),
            "context_reliance": str(classification.context_reliance),
            "retrieval_budget": orchestrator.max_retrieval_tokens,
            "recent_budget": orchestrator.recent_token_budget,
        }
    finally:
        db.close()


def ollama_unload(model: str) -> None:
    """Evict a model from VRAM by asking Ollama for a zero keep_alive.

    ⚑ WHY THIS IS NECESSARY. Ingestion needs the small background model; answering
    needs the 17.2 GB answerer. Left to Ollama's own policy the answerer stays
    resident and starves the background model -- measured: qwen3:4b-instruct-bg
    held only 0.3 GB of a 2.5 GB footprint, ran largely off-GPU, and blew
    codex_extractor's 30 s timeout on every real turn. The visible result was
    `triplet_parsing_failed: Request timed out` and a knowledge graph with ZERO
    entities, while ingestion still reported success.

    So the run drives residency explicitly, which is this project's stated position
    anyway: keep_alive is ICE's policy to set, not a host default to inherit.
    """
    import urllib.error
    import urllib.request
    ollama_base = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434").rstrip("/")
    payload = json.dumps({"model": model, "keep_alive": 0, "stream": False}).encode()
    req = urllib.request.Request(
        f"{ollama_base}/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass  # eviction is an optimisation; never fail the run over it


def _ollama_loaded_models() -> list[str]:
    import urllib.request

    base = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434").rstrip("/")
    with urllib.request.urlopen(f"{base}/api/ps", timeout=15) as response:
        body = json.loads(response.read())
    return [
        model["name"] for model in body.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    ]


def ensure_ollama_background_resident(model: str = EXPECTED_BACKGROUND_MODEL) -> None:
    """Make the final harness use exactly one resident Ollama background model.

    v2's shared-mode factory can reuse whichever model happens to be loaded. That
    is useful for interactive ICE, but it makes a benchmark depend on the user's
    previous chat and can select the cloud/local answerer instead of the measured
    background model. Evict all other Ollama models, then load the pinned model
    with an infinite keep-alive. The vLLM path remains diagnostic-only and does not
    call this function.
    """
    loaded = _ollama_loaded_models()
    for loaded_model in loaded:
        if loaded_model != model:
            ollama_unload(loaded_model)

    import urllib.request

    base = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434").rstrip("/")
    payload = json.dumps({
        "model": model,
        "prompt": "Reply with the single word: ok",
        "stream": False,
        "keep_alive": -1,
        "options": {"num_predict": 16},
    }).encode()
    request = urllib.request.Request(
        f"{base}/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read())
    if not (result.get("response") or "").strip():
        raise RuntimeError(
            f"Ollama background model {model!r} returned empty content during "
            "residency preflight"
        )
    if model not in _ollama_loaded_models():
        raise RuntimeError(
            f"Ollama background model {model!r} did not remain resident after "
            "the residency preflight"
        )


def preflight(sample_pair: tuple[str, str] | None = None) -> tuple[bool, dict]:
    """Refuse to start if the background model is dead.

    ⚑ THIS EXISTS BECAUSE THE FAILURE IS SILENT. The tag ships `get_bg_client()`
    hardcoded to an SGLang server on :8001 (the Experiment 3 ablation config). With
    nothing on that port, every background call failed *and every call site
    swallowed it*: summaries were written as empty strings, codex extraction
    returned zero triplets, and the run completed with plausible row counts. The
    resulting number would have described ICE with no knowledge graph and no
    summarisation, labelled as ICE.

    So the pipeline is exercised for real before any instance is processed --
    CLAUDE.md's "prove each part twice", and its rule that a fallback firing on
    100% of calls is an outage wearing resilience as a costume.
    """
    from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
    from src.workers.post_flight import generate_summary
    from src.workers.codex_extractor import extract_triplets

    info: dict = {}
    model = get_bg_model_name()
    info["background_model"] = model
    if model != EXPECTED_BACKGROUND_MODEL:
        print(
            f"⛔ PREFLIGHT: v2 resolved background model {model!r}, expected "
            f"{EXPECTED_BACKGROUND_MODEL!r}. Refusing a mixed-model run.",
            file=sys.stderr,
        )
        return False, info

    try:
        r = get_bg_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=16,
        )
        reply = (r.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"⛔ PREFLIGHT: background model unreachable ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return False, info
    if not reply:
        print(f"⛔ PREFLIGHT: background model {model!r} returned EMPTY at a small\n"
              f"   token budget. Thinking models do this -- they spend the budget\n"
              f"   reasoning. Summaries and codex extraction would silently produce\n"
              f"   nothing. Use a non-thinking instruct model.", file=sys.stderr)
        return False, info
    info["probe_reply"] = reply

    # Two SEPARATE checks, because they fail for different reasons and conflating
    # them produced two wrong verdicts in a row:
    #   (1) CAPABILITY -- fact-dense text must yield triplets. A short toy sample
    #       is right here; the question is whether extraction works at all.
    #   (2) ENDURANCE  -- a real, long LongMemEval turn must complete without
    #       raising. ZERO triplets is a legitimate outcome for it: haystack turns
    #       are largely generic chatter (the first oracle turn is a fox-chicken-
    #       grain river puzzle), so demanding facts from one is simply wrong.
    # Version one checked (1) with a toy sample only, passed, and then timed out on
    # every real turn. Version two checked (2) alone and failed a healthy system.
    sample_p = "I moved to Berlin last March and started working at Vertex Labs."
    sample_r = "Got it -- Berlin since March, and you're at Vertex Labs now."
    try:
        triplets = extract_triplets(f"User: {sample_p}\n\nAssistant: {sample_r}",
                                    topic_tags=["Personal_Life"])
    except Exception as exc:  # noqa: BLE001
        print(f"⛔ PREFLIGHT: extract_triplets raised ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return False, info
    info["preflight_triplets"] = len(triplets or [])
    if not triplets:
        print(f"⛔ PREFLIGHT: codex extraction produced ZERO triplets on a sentence\n"
              f"   with obvious facts in it. The knowledge graph would stay empty\n"
              f"   for the whole run and nothing downstream would say so.",
              file=sys.stderr)
        return False, info
    observed = {
        (
            str(t.get("subject", "")).strip().lower(),
            str(t.get("relation", "")).strip().lower(),
            str(t.get("object", "")).strip().lower(),
        )
        for t in triplets if isinstance(t, dict)
    }
    expected = {
        ("user", "lives_in", "berlin"),
        ("user", "works_at", "vertex labs"),
    }
    info["preflight_expected_triplets"] = len(expected & observed)
    if not expected.issubset(observed):
        print(
            "⛔ PREFLIGHT: extraction returned JSON but failed the grounded "
            "direction control.\n"
            f"   expected both {sorted(expected)!r}\n"
            f"   observed {sorted(observed)!r}\n"
            "   A reachable model that reverses facts would poison the whole store.",
            file=sys.stderr,
        )
        return False, info

    try:
        summary = generate_summary(sample_p, sample_r)
    except Exception as exc:  # noqa: BLE001
        print(f"⛔ PREFLIGHT: generate_summary raised ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return False, info
    if not (summary or "").strip():
        print("⛔ PREFLIGHT: generate_summary returned an empty string.", file=sys.stderr)
        return False, info
    if "<think>" in summary.lower():
        print("⛔ PREFLIGHT: generate_summary leaked reasoning markup into memory.",
              file=sys.stderr)
        return False, info
    info["preflight_summary_chars"] = len(summary)

    # (2) ENDURANCE on a real turn. Measured: 8.1 s on 1,543 chars but 42.8 s on
    # 3,512, which is why the tag's 30 s extraction timeout had to be raised. Zero
    # triplets here is fine; an exception or a timeout is not.
    if sample_pair:
        long_p, long_r = sample_pair
        info["endurance_sample_chars"] = len(long_p) + len(long_r)
        t0 = time.time()
        try:
            long_triplets = extract_triplets(f"User: {long_p}\n\nAssistant: {long_r}",
                                             topic_tags=["Personal_Life"])
        except Exception as exc:  # noqa: BLE001
            print(f"⛔ PREFLIGHT: extraction failed on a real {info['endurance_sample_chars']}-char\n"
                  f"   turn ({type(exc).__name__}: {exc}). Long turns would silently\n"
                  f"   contribute nothing to the graph for the whole run.", file=sys.stderr)
            return False, info
        info["endurance_seconds"] = round(time.time() - t0, 1)
        info["endurance_triplets"] = len(long_triplets or [])
        print(f"  endurance OK: {info['endurance_sample_chars']} chars -> "
              f"{info['endurance_triplets']} triplets in {info['endurance_seconds']}s")

    print(f"  preflight OK: bg={model} triplets={len(triplets)} "
          f"summary={len(summary)} chars")
    return True, info


def store_matches(layout, SessionLocal, models) -> bool:
    """Is the store ALREADY holding exactly this instance's haystack?

    ⚑ This is the mature harness's restart guard, adapted. There, a checkpoint
    was atomic: it checked whether the checkpoint's last turn was already in the
    DB and, if so, skipped replay AND the background workers, so the simulated
    decay loop could never compound across restarts.

    LME cannot compound decay -- every instance starts from a wipe and no decay
    is simulated -- but the same guard buys resumability: if a run died during the
    ANSWER pass, the ~500-turn ingestion is still sitting in the store and should
    not be paid for twice. We require an exact count and no foreign rows, so a
    partial ingestion is never mistaken for a complete one.
    """
    from sqlalchemy import func

    Conversation = models["Conversation"]
    EpisodicMemory = models["EpisodicMemory"]
    expected_counts = {
        spec["conversation_id"]: len(spec["pairs"])
        for spec in layout["sessions"] if spec["pairs"]
    }
    expected_conversations = {
        layout["query_conversation_id"],
        *(spec["conversation_id"] for spec in layout["sessions"]),
    }
    db = SessionLocal()
    try:
        actual_counts = {
            cid: count for cid, count in db.query(
                EpisodicMemory.conversation_id, func.count(EpisodicMemory.id)
            ).group_by(EpisodicMemory.conversation_id).all()
        }
        actual_conversations = {
            cid for (cid,) in db.query(Conversation.id).all()
        }
        return (actual_counts == expected_counts
                and actual_conversations == expected_conversations
                and sum(actual_counts.values()) > 0)
    finally:
        db.close()


def _to_pairs(session: list[dict]) -> list[tuple[str, str]]:
    """Fold a session's [{role, content}, ...] into (user, assistant) pairs.
    An unpaired trailing user turn is kept with an empty response rather than
    dropped -- dropping it would silently delete evidence."""
    pairs, pending = [], None
    for turn in session:
        role, content = turn.get("role"), turn.get("content") or ""
        if role == "user":
            if pending is not None:
                pairs.append((pending, ""))
            pending = content
        elif role == "assistant":
            pairs.append((pending if pending is not None else "", content))
            pending = None
    if pending is not None:
        pairs.append((pending, ""))
    return pairs


def _parse_lme_date(raw: str | None) -> datetime:
    """LongMemEval dates look like '2023/05/20 (Sat) 02:21'."""
    if not raw:
        return datetime.now(timezone.utc)
    cleaned = " ".join(p for p in raw.split() if not p.startswith("("))
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def instance_layout(inst: dict, phase: str) -> dict:
    """Map one LongMemEval instance onto v2's real conversation model.

    A benchmark instance is isolated by a full store wipe. Inside it, each
    timestamped history session is a distinct auto-scoped ICE conversation, and
    the question is asked in a fresh empty auto-scoped conversation. This is the
    production shape for cross-session recall; flattening all sessions into one
    conversation silently measures within-chat memory instead.
    """
    qid = inst["question_id"]
    sessions = inst.get("haystack_sessions") or []
    dates = inst.get("haystack_dates") or []
    session_ids = inst.get("haystack_session_ids") or []
    specs = []
    for index, messages in enumerate(sessions):
        session_id = (session_ids[index] if index < len(session_ids)
                      else f"session-{index}")
        raw_date = dates[index] if index < len(dates) else None
        specs.append({
            "source_index": index,
            "session_id": session_id,
            "conversation_id": uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"lme:{phase}:{qid}:session:{index}:{session_id}",
            ),
            "timestamp": _parse_lme_date(raw_date),
            "raw_date": raw_date,
            "pairs": _to_pairs(messages),
        })

    # Oracle files do not guarantee chronological ordering. Online memory must
    # ingest sessions in the order they happened for update/version semantics.
    specs.sort(key=lambda spec: (spec["timestamp"], spec["source_index"]))
    return {
        "adapter_version": ADAPTER_VERSION,
        "query_conversation_id": uuid.uuid5(
            uuid.NAMESPACE_DNS, f"lme:{phase}:{qid}:query"
        ),
        "sessions": specs,
    }


def auto_query_context(query_conversation_id) -> tuple[str, dict]:
    """Return the identifier/scope shape v2 production uses for auto memory."""
    return str(query_conversation_id), {}


def main() -> int:
    global _active_stage, _signal_count, _stop_requested
    _stop_requested = False
    _signal_count = 0
    _active_stage = "starting"

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--answer-profile",
        default="local-gemma26",
        help="pinned profile from cloud_provider.py; credentials are read only "
             "from the profile's environment-variable name",
    )
    ap.add_argument(
        "--background-provider", choices=("ollama", "vllm"), default="ollama",
        help="ollama is the accepted path; vllm is retained only to reproduce "
             "the rejected parity experiment on localhost:8002",
    )
    ap.add_argument(
        "--answer-max-tokens", type=int, default=ANSWER_MAX_TOKENS,
        help="visible+reasoning output cap; cloud wrapper pins 4096",
    )
    ap.add_argument("--plan-only", action="store_true",
                    help="print what would run and exit -- touches nothing")
    args = ap.parse_args()

    from cloud_provider import (
        MAIN_ENV, PROFILES, ProviderAccessError, TextGenerator, get_profile,
        load_selected_env,
    )

    if args.answer_profile not in PROFILES:
        print(f"⛔ unknown --answer-profile {args.answer_profile!r}; choose one of: "
              f"{', '.join(sorted(PROFILES))}", file=sys.stderr)
        return 2
    answer_profile = get_profile(args.answer_profile)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    spec = PHASES[args.phase]
    corpus_path = DATA_DIR / spec["corpus"]
    if not corpus_path.exists():
        print(f"⛔ corpus missing: {corpus_path}\n   run: uv run python "
              f"{HARNESS_DIR / 'fetch_dataset.py'}", file=sys.stderr)
        return 1

    corpus = json.loads(corpus_path.read_text())
    instances = select_instances(corpus, spec["select"], args.limit, args.seed)
    out_dir = args.out / args.phase
    answers_dir = out_dir / "answers"

    # Resume at CONDITION granularity: an instance counts as done only when every
    # condition has an answer. A run that died between conditions resumes into the
    # missing one instead of redoing the whole instance.
    def _completed(qid: str) -> bool:
        path = answers_dir / f"{qid}.json"
        if not path.exists():
            return False
        rec = _read_json(path)
        if rec is None:
            return False  # unreadable == not done; it will be rebuilt
        return (rec.get("adapter_version") == ADAPTER_VERSION
                and rec.get("status") == "complete"
                and all(answer_complete_for_profile(
                    (rec.get("answers") or {}).get(cond), answer_profile)
                        for cond in CONDITIONS))

    done = {x["question_id"] for x in instances if _completed(x["question_id"])}
    todo = [x for x in instances if x["question_id"] not in done]
    turns_todo = sum(sum(len(s) for s in x["haystack_sessions"]) for x in todo)

    print(f"phase={args.phase}  corpus={spec['corpus']}")
    print(f"  adapter {ADAPTER_VERSION}")
    print(f"  selected {len(instances)}  done {len(done)}  remaining {len(todo)}")
    print(f"  ~{turns_todo:,} turns to ingest")
    if args.plan_only:
        for x in todo[:10]:
            n = sum(len(s) for s in x["haystack_sessions"])
            print(f"    {x['question_id']:12} {x['question_type']:26} {n:4d} turns")
        if len(todo) > 10:
            print(f"    ... and {len(todo) - 10} more")
        return 0

    answers_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "MANIFEST.json"
    existing_manifest = _read_json(manifest_path) if manifest_path.exists() else None
    if existing_manifest:
        existing_env = existing_manifest.get("environment") or {}
        existing_answerer = existing_env.get("answerer") or {}
        manifest_mismatch = []
        if existing_manifest.get("adapter_version") not in (None, ADAPTER_VERSION):
            manifest_mismatch.append("adapter version")
        if existing_answerer and (
            existing_answerer.get("profile") != answer_profile.name
            or existing_answerer.get("model") != answer_profile.model
            or existing_answerer.get("endpoint") != answer_profile.endpoint
        ):
            manifest_mismatch.append("answerer profile/model/endpoint")
        if not existing_answerer and answer_profile.name != "local-gemma26":
            manifest_mismatch.append("missing cloud answerer identity")
        if existing_env.get("background_provider") not in (None, args.background_provider):
            manifest_mismatch.append("background provider")
        if manifest_mismatch:
            print(
                "⛔ existing run manifest does not match this invocation: "
                + ", ".join(manifest_mismatch) + ".\n"
                "   Refusing to mix artifacts; choose a fresh --out root.",
                file=sys.stderr,
            )
            return 5

    mismatched = [
        x["question_id"] for x in instances
        if (answers_dir / f"{x['question_id']}.json").exists()
        and ((_read_json(answers_dir / f"{x['question_id']}.json") or {}).get(
            "adapter_version") != ADAPTER_VERSION)
    ]
    if mismatched:
        print(f"⛔ {len(mismatched)} answer artifact(s) use the invalid flattened-session adapter.\n"
              f"   Refusing to reuse or overwrite them. Run the reversible repair first:\n"
              f"   uv run python {HARNESS_DIR / 'repair_invalid_adapter.py'} "
              f"--phase {args.phase} --apply", file=sys.stderr)
        return 4
    profile_mismatched = []
    for instance in instances:
        path = answers_dir / f"{instance['question_id']}.json"
        rec = _read_json(path) if path.exists() else None
        answers = (rec or {}).get("answers") or {}
        if answers and any(
            answer is not None and not answer_identity_matches(answer, answer_profile)
            for answer in answers.values()
        ):
            profile_mismatched.append(instance["question_id"])
    if profile_mismatched:
        print(
            f"⛔ {len(profile_mismatched)} answer artifact(s) were produced by a "
            f"different answer profile.\n"
            f"   Refusing to mix or overwrite them. Select a fresh --out root for "
            f"{answer_profile.name}.\n"
            f"   First ids: {', '.join(profile_mismatched[:5])}",
            file=sys.stderr,
        )
        return 5
    if not todo:
        print("  nothing to do -- all selected instances already have answers.")
        return 0

    # Imports deferred until after --plan-only so the planner needs no DB or GPU.
    # Cloud credentials live in main's .env; load ONLY their selected names so
    # v3-only settings cannot poison v2's extra-forbid Settings object.
    load_selected_env()
    if args.background_provider == "ollama":
        # v2 reads OLLAMA_BASE_URL from its Settings. The shell-facing
        # OLLAMA_HOST_URL name is deliberately translated here so the harness
        # and the v2 client cannot silently probe different Ollama servers.
        os.environ["OLLAMA_BASE_URL"] = os.environ.get(
            "OLLAMA_HOST_URL", "http://localhost:11434"
        ).rstrip("/")
        try:
            ensure_ollama_background_resident()
        except Exception as exc:  # noqa: BLE001 - preflight must fail loudly
            print(
                f"⛔ Ollama background residency preflight failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 3
    if args.background_provider == "vllm":
        os.environ["BACKGROUND_MODEL_MODE"] = "dedicated"
    from src.api.config import settings
    from src.api.db import SessionLocal
    from src.memory.models import Base, Conversation, EpisodicMemory, MemorySlot
    from src.classifier.classifier import PyTorchClassifier

    env = capture_env(settings.database_url)
    env["answerer"] = answer_profile.metadata()
    env["answer_generation"] = {
        "temperature_requested": ANSWER_TEMPERATURE,
        "temperature_supported": answer_profile.supports_temperature,
        "max_output_tokens": args.answer_max_tokens,
    }
    env["background_provider"] = args.background_provider
    if args.background_provider == "vllm":
        env["background_actual_model"] = os.environ.get(
            "LME_BG_ACTUAL_MODEL", "unrecorded"
        )
        env["background_revision"] = os.environ.get(
            "LME_BG_REVISION", "unrecorded"
        )
    _atomic_write_json(manifest_path, {"phase": args.phase,
                                                   "selection": spec["select"],
                                                   "seed": args.seed,
                                                   "n_selected": len(instances),
                                                   "adapter_version": ADAPTER_VERSION,
                                                   "environment": env})
    # ⚑ Paths MUST come from settings, exactly as the mature harness does
    # (run_mature_experiment.py:342). PyTorchClassifier's own default is a stale
    # `models/classifier/ice_classifier.pt` — a different, older checkpoint — so
    # constructing it bare silently evaluates the wrong classifier.
    classifier = PyTorchClassifier(
        model_path=settings.classifier_model_path,
        schema_path=settings.label_schema_path,
    )
    embedder = classifier.embedder
    dim = len(embedder.encode("dimension probe", convert_to_tensor=False))
    if dim != 384:
        print(f"⛔ embedder returned {dim} dims, expected 384.\n"
              f"   v2 stores Vector(384); `main` embeds at 1024 with the SAME model\n"
              f"   name. You are almost certainly running under main's venv.\n"
              f"   Run this from the worktree so uv resolves v2's lockfile.",
              file=sys.stderr)
        return 2
    print(f"  embedder OK: {dim} dims")

    # Longest turn among the first few instances -- a worst-ish case, so a pass
    # here means the real workload is within reach rather than merely a toy.
    _cand = [pr for x in todo[:3] for sess in x["haystack_sessions"]
             for pr in _to_pairs(sess)]
    _sample = max(_cand, key=lambda pr: len(pr[0]) + len(pr[1])) if _cand else None
    # Evict the answerer BEFORE probing. A leftover 17.2 GB resident from an
    # earlier pass starves the background model and preflight fails for the wrong
    # reason -- measured: extraction times out with the answerer loaded, and
    # returns 3 triplets in 8.1s once the card is clear.
    if answer_profile.name == "local-gemma26":
        ollama_unload(SINGLE_MODEL)
    ok, pf = preflight(_sample)
    env["preflight"] = pf
    _atomic_write_json(manifest_path, {"phase": args.phase,
                                                   "selection": spec["select"],
                                                   "seed": args.seed,
                                                   "n_selected": len(instances),
                                                   "adapter_version": ADAPTER_VERSION,
                                                   "environment": env})
    if not ok:
        print("\n   Nothing has been written. Fix the above and re-run.", file=sys.stderr)
        return 3
    print()

    models = {"Conversation": Conversation, "EpisodicMemory": EpisodicMemory,
              "MemorySlot": MemorySlot}
    generator = TextGenerator(answer_profile)
    started = time.time()
    durations: list[float] = []
    fatal_provider_error = None

    for i, inst in enumerate(todo, 1):
        if _stop_requested:
            print("  stopped by request; progress is on disk.")
            break
        qid = inst["question_id"]
        _active_stage = f"preparing {qid}"
        n_turns = sum(len(s) for s in inst["haystack_sessions"])
        t0 = time.time()
        eta = (sum(durations) / len(durations) * (len(todo) - i + 1)) if durations else None
        print(f"[{i:4d}/{len(todo)}] {qid:12} {inst['question_type']:26} "
              f"{n_turns:4d} turns  elapsed {_fmt(time.time() - started)}"
              f"{'  eta ' + _fmt(eta) if eta else ''}", flush=True)

        try:
            layout = instance_layout(inst, args.phase)
            query_cid = layout["query_conversation_id"]
            rec_path = answers_dir / f"{qid}.json"
            rec = {}
            if rec_path.exists():
                try:
                    rec = json.loads(rec_path.read_text())
                except (json.JSONDecodeError, OSError):
                    rec = {}

            expected = sum(len(_to_pairs(s)) for s in inst["haystack_sessions"])
            if store_matches(layout, SessionLocal, models):
                # The haystack from a previous, interrupted pass is still loaded.
                # Reuse it rather than pay ~500 turns again for a failed answer.
                stored = expected
                print(f"           store already holds this haystack "
                    f"({expected} pairs) — skipping ingest", flush=True)
            else:
                _active_stage = f"ingesting {qid}"
                # Give ingestion the whole card: evict the answerer so the small
                # background model is fully resident. Without this, extraction runs
                # off-GPU and times out silently (see ollama_unload).
                ollama_unload(SINGLE_MODEL)

                # Fresh state per instance. Wipe FIRST, so an instance interrupted
                # last run cannot leak its rows into this one.
                wipe_store(SessionLocal, Base)
                db = SessionLocal()
                try:
                    for session_spec in layout["sessions"]:
                        db.add(Conversation(
                            id=session_spec["conversation_id"],
                            memory_scope_type="auto",
                        ))
                    db.add(Conversation(
                        id=query_cid, memory_scope_type="auto"
                    ))
                    db.commit()
                finally:
                    db.close()

                stored = ingest_instance(layout, classifier, embedder,
                                         SessionLocal, models,
                                         progress_prefix=f"[{i}/{len(todo)}] {qid}")

                # Clustering once, so cluster-scoped retrieval has something to
                # scope to. No decay simulation -- see the module docstring.
                from src.workers.clustering import cluster_turns, merge_similar_clusters
                _active_stage = f"clustering {qid}"
                cluster_turns()
                merge_similar_clusters()
                # `stored` counts user/assistant PAIRS, n_turns counts raw turns;
                # printing "18/36" read as a 50% loss when it is a complete replay.
                print(f"           ingested {stored} pairs from {n_turns} turns "
                      f"in {_fmt(time.time() - t0)}", flush=True)

            # ⚑ A stop during ingestion ABANDONS the instance -- it must never fall
            # through to answering. Without this the runner broke out of the ingest
            # loop, answered against a PARTIAL haystack, and wrote status="complete":
            # a wrong number that looked finished. Caught by the SIGINT test.
            #
            # Nothing is written, so the instance is simply redone next run. If the
            # haystack happens to be complete, store_matches() reuses it and the
            # work is not lost.
            if _stop_requested:
                short = stored < expected
                print(f"           stopped during ingestion ({stored}/{expected} pairs)"
                      f" — instance abandoned, no answer written."
                      f"{' Store is complete and will be reused.' if not short else ''}",
                      flush=True)
                break

            rec.update({
                "adapter_version": ADAPTER_VERSION,
                "question_id": qid,
                "question_type": inst["question_type"],
                "is_abstention": qid.endswith("_abs"),
                "question": inst["question"],
                "question_date": inst.get("question_date"),
                "reference_answer": inst.get("answer"),
                "query_conversation_id": str(query_cid),
                "history_conversation_ids": [
                    str(session_spec["conversation_id"])
                    for session_spec in layout["sessions"]
                ],
                "memory_scope_type": "auto",
                "turns_stored": stored,
                "turns_in_haystack": n_turns,
                "status": "ingested",
            })
            rec.setdefault("answers", {})
            _atomic_write_json(rec_path, rec)

            # History size is provenance only. v2's production budget is derived
            # from the NEW query conversation, which has zero turns.
            total_tokens = sum(
                estimate_tokens(p + " " + r)
                for s in inst["haystack_sessions"] for p, r in _to_pairs(s)
            )
            rec["history_pairs"] = expected
            rec["history_tokens_estimate"] = total_tokens
            _atomic_write_json(rec_path, rec)
            # Ingestion is done; hand the card back to the answerer.
            missing_conditions = [
                c for c in CONDITIONS
                if not answer_complete_for_profile(
                    rec["answers"].get(c), answer_profile)
            ]
            if (missing_conditions
                    and args.background_provider == "ollama"
                    and answer_profile.name == "local-gemma26"):
                from src.workers.bg_client_factory import get_bg_model_name
                ollama_unload(get_bg_model_name())

            for cond in CONDITIONS:
                if answer_complete_for_profile(
                        rec["answers"].get(cond), answer_profile):
                    continue  # already answered in an earlier pass
                a0 = time.time()
                _active_stage = f"answering {cond} for {qid}"
                print(
                    f"           {_active_stage} with {answer_profile.model} "
                    f"(timeout {answer_profile.timeout_seconds:g}s, "
                    f"retries {answer_profile.max_retries})",
                    flush=True,
                )
                rec["answers"][cond] = answer_instance(
                    inst, query_cid, cond, classifier, embedder, SessionLocal,
                    generator, models, args.answer_max_tokens,
                    provider_session_id=str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"ice-lme-answer:{args.phase}:{qid}:{cond}:"
                        f"{answer_profile.name}",
                    )))
                _atomic_write_json(rec_path, rec)  # persist per condition
                print(f"           {cond:10} {rec['answers'][cond]['fragments']:3d} frags "
                      f"{rec['answers'][cond]['tokens_injected']:6d} tok "
                      f"{_fmt(time.time() - a0)}", flush=True)

            rec["status"] = "complete"
            rec["total_seconds"] = round(time.time() - t0, 1)
            _atomic_write_json(rec_path, rec)
            durations.append(time.time() - t0)
            _active_stage = "between instances"
        except ProviderAccessError as exc:
            _active_stage = "between instances"
            fatal_provider_error = exc
            print(
                f"           ⛔ PROVIDER {exc.kind.upper()}: {exc}\n"
                "           Stopping now; completed answer files remain valid.\n"
                f"           Replace PROBE_API_KEY in {MAIN_ENV}, then rerun the "
                "same command.",
                flush=True,
            )
            with (out_dir / "failures.log").open("a") as fh:
                fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{qid}\t"
                         f"ProviderAccessError[{exc.kind}]\t{exc}\n")
            break
        except Exception as exc:  # noqa: BLE001 -- one bad instance must not end the run
            _active_stage = "between instances"
            # The ingested/partial-condition record remains resumable. Record one
            # isolated failure and continue; provider-wide failures stop above.
            print(f"           ⛔ FAILED: {type(exc).__name__}: {exc}", flush=True)
            with (out_dir / "failures.log").open("a") as fh:
                fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{qid}\t"
                         f"{type(exc).__name__}\t{exc}\n")

    _active_stage = "finished"
    remaining = len([x for x in instances if not _completed(x["question_id"])])
    print(f"\ndone this pass. {remaining} instance(s) still outstanding.")
    print(f"artifacts: {out_dir}")
    if remaining:
        print("re-run the same command to continue.")
    return 7 if fatal_provider_error else 0


if __name__ == "__main__":
    sys.exit(main())
