"""Z2 S0: where the replay write path spends its time (spec Z2_EVALUATION §2, S0).

Drives the production replay instrument (`scripts/z1/seed_v3.py`) on a prefix of
history B, re-timed onto the declared Z2 clock (`clock.py`) so the maintenance
cadence matches what Z2 will run. Nothing in `src/` or `seed_v3` changes: the
profiler wraps functions from outside and records, per call, a nested stage
path and its wall time. Every model call carries the stage that made it, its
token counts and the provider's own load/prompt/eval split. The memory trace
is seed_v3's own (`seed.jsonl`); the profile is a sidecar (`profile.jsonl`)
appended after every top-level stage, so a killed run keeps what it measured.

Run inside a dedicated scratch database (`--create-db` makes it, refusing an
existing one), resumable with `--resume`:

    uv run --no-sync python experiments/z2/profile_write_path.py \
        --out logs/z2-s0-speed-profile-<date>-r1 --limit 150 --create-db

Wrapper cost is a perf_counter pair per call; what it does not time (SQL inside
a stage, Python glue) lands in each stage's self time.
"""

import argparse
import functools
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-s0-write-profile-1"
HISTORY_SLUG = {"B": "bb558b5f"}


class Profiler:
    """Nested wall-clock stages; top-level stages are flushed to a JSONL sidecar."""

    def __init__(self, sink):
        self.sink = sink
        self.local = threading.local()
        self.seq = 0
        self.turn_key = None
        self.last_provider = None

    def _stack(self):
        if not hasattr(self.local, "stack"):
            self.local.stack = []
        return self.local.stack

    @contextmanager
    def stage(self, name, **info):
        stack = self._stack()
        node = {"stage": name, "children": [], "calls": [], **info}
        stack.append(node)
        started = time.perf_counter()
        error = None
        try:
            yield node
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            node["seconds"] = round(time.perf_counter() - started, 4)
            if error:
                node["error"] = error
            stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                self.seq += 1
                node.update(seq=self.seq, turn_key=self.turn_key, wall_end=time.time())
                self.sink.write(json.dumps(node, default=str) + "\n")
                self.sink.flush()

    def model_call(self, record):
        stack = self._stack()
        if stack:
            stack[-1]["calls"].append(record)
        else:
            self.sink.write(json.dumps({"stage": "unattributed_model_call", "calls": [record],
                                        "seq": None, "turn_key": self.turn_key}) + "\n")
            self.sink.flush()

    def wrap(self, owner, attribute, name, **info_from_args):
        original = getattr(owner, attribute)
        profiler = self

        @functools.wraps(original)
        def timed(*args, **kwargs):
            info = {k: f(args, kwargs) for k, f in info_from_args.items()}
            with profiler.stage(name, **info):
                return original(*args, **kwargs)

        return patch.object(owner, attribute, timed)


def install(profiler, stack):
    """Every patch the profile needs; returns nothing, enters each into *stack*."""
    import sentence_transformers
    from scripts.z1 import production_parity, seed_v3
    from src.api import source_proof
    from src.memory import facts, pattern_support, summary_support, support
    from src.retrieval import ner_utils
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    from src.workers import (bg_client_factory, bg_native_transport, conversation_summary,
                             fact_extractor, post_flight, procedural_extractor)

    def store_key(args, kwargs):
        key = args[5] if len(args) > 5 else kwargs.get("idempotency_key")
        profiler.turn_key = key
        return key

    def words(args, kwargs):
        return sum(len(str(a).split()) for a in args[2:4]) if len(args) > 3 else None

    patches = [
        # Per historical turn, in the order seed_v3 calls them.
        profiler.wrap(production_parity, "build", "preflight.build"),
        profiler.wrap(seed_v3, "prepare_with_trace", "preflight.retrieve"),
        # Production work inside seed_v3's tracing helper; the helper's own
        # capture is what remains in preflight.retrieve's self time.
        profiler.wrap(production_parity, "prepare", "pp.prepare"),
        profiler.wrap(HybridRetrievalOrchestrator, "retrieve", "orchestrator.retrieve"),
        profiler.wrap(seed_v3, "_store_turn", "write.store", key=store_key, words=words),
        profiler.wrap(post_flight, "evaluate_turn", "post_flight"),
        profiler.wrap(seed_v3, "due_maintenance", "maintenance"),
        profiler.wrap(seed_v3, "probe_observation", "probe.observe"),
        # Inside post-flight.
        profiler.wrap(post_flight, "extract_key_terms", "key_terms"),
        profiler.wrap(post_flight, "generate_summary", "summary"),
        profiler.wrap(post_flight, "verify_representations", "verify_representations"),
        profiler.wrap(post_flight, "run_chunk_turn", "chunk"),
        profiler.wrap(post_flight, "extract_facts", "facts"),
        profiler.wrap(post_flight, "extract_procedural", "procedural"),
        profiler.wrap(fact_extractor, "_claim_record", "facts.own_claim_pass"),
        profiler.wrap(facts, "write_turn_facts", "facts.write"),
        profiler.wrap(facts, "candidates", "facts.candidates"),
        profiler.wrap(fact_extractor, "_ask", "facts.reconcile_call"),
        profiler.wrap(fact_extractor, "_mentions", "facts.mentions"),
        # Imported by name into several modules: patch every binding.
        *[profiler.wrap(module, "generate_claims", "claims.generate")
          for module in (post_flight, fact_extractor, conversation_summary)],
        *[profiler.wrap(module, "verify_assertions", "claims.verify")
          for module in (summary_support, fact_extractor, conversation_summary)],
        *[profiler.wrap(module, "score_pairs", "nli.score_pairs",
                        pairs=lambda a, k: len(a[0]) if a else None)
          for module in (support, source_proof, pattern_support)],
        profiler.wrap(ner_utils, "_extract_background", "ner.background"),
        profiler.wrap(sentence_transformers.SentenceTransformer, "encode", "embed.encode"),
        profiler.wrap(sentence_transformers.CrossEncoder, "predict", "rerank.predict",
                      pairs=lambda a, k: len(a[1]) if len(a) > 1 else None),
    ]

    import importlib
    from src.workers.runtime import JOBS
    for job in seed_v3.MEMORY_JOBS:
        module, function = JOBS[job].path.split(":")
        patches.append(profiler.wrap(importlib.import_module(module), function, f"job.{job}"))

    original_timings = bg_client_factory.native_call_timings

    def stash_timings(data):
        result = original_timings(data)
        profiler.last_provider = result
        return result

    original_call = bg_native_transport.chat_completion

    def timed_call(**kwargs):
        started = time.perf_counter()
        profiler.last_provider = None
        record = {"model": kwargs.get("model"), "max_tokens": kwargs.get("max_tokens")}
        try:
            result = original_call(**kwargs)
            usage = getattr(result, "usage", None)
            record.update(prompt_tokens=getattr(usage, "prompt_tokens", None),
                          completion_tokens=getattr(usage, "completion_tokens", None),
                          finish=result.choices[0].finish_reason)
            return result
        except BaseException as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["wall"] = round(time.perf_counter() - started, 4)
            record.update(profiler.last_provider or {})
            profiler.model_call(record)

    patches += [patch.object(bg_client_factory, "native_call_timings", stash_timings),
                patch.object(bg_native_transport, "chat_completion", timed_call)]
    for item in patches:
        stack.enter_context(item)
    del procedural_extractor  # imported so its module is loaded before patching post_flight


def retime(conversations, slug, history):
    """Replace the synthetic timestamps of *slug* with the declared Z2 clock."""
    from datetime import datetime
    from experiments.z2.clock import declared_times
    rows = conversations[slug]
    if {r["ts_provenance"] for r in rows} != {"synthetic_raw_import"}:
        raise ValueError("only synthetic timestamps may be re-timed (D3)")
    times = declared_times(history, len(rows), rows[0]["timestamp"])
    for row, when in zip(rows, times):
        row["timestamp"] = when
    assert all(isinstance(r["timestamp"], datetime) for r in rows)
    return times


# Scratch databases experiments may create and wipe: Z2's and FINAL's (run_final names them
# ice_final_<run>_<history>; the guard admitted only ice_z2_, so every FINAL replay refused — caught by
# the GPU day's LoCoMo trial, 2026-10-30). Never the live store.
SCRATCH_PREFIXES = ("ice_z2_", "ice_final_")


def create_database(name):
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from src.api.config import settings
    from src.memory.models import Base
    source = make_url(settings.database_url)
    if name == source.database or name == "ice_db" or not name.startswith(SCRATCH_PREFIXES):
        raise ValueError("scratch database must be named ice_z2_* or ice_final_* and differ from the live store")
    admin = create_engine(source.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        if conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar():
            raise RuntimeError(f"{name} already exists; use --resume or a new name")
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    engine = create_engine(source.set(database=name))
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    engine.dispose()
    return source.set(database=name).render_as_string(hide_password=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="private run directory under logs/")
    parser.add_argument("--history", choices=sorted(HISTORY_SLUG), default="B")
    parser.add_argument("--limit", type=int, required=True, help="replay pairs 1..limit")
    parser.add_argument("--database", default="ice_z2_s0_profile_r1")
    parser.add_argument("--create-db", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-failure-policy", choices=("strict", "continue"), default="strict",
                        help="continue: a failed model worker is recorded as degraded and the replay goes on")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if os.environ.get("ICE_TEST_DATABASE") != args.database:
            raise RuntimeError("child started without its scratch database")
        return run_profile(args)
    from scripts.z1.private_logs import under_logs
    if not under_logs(args.out):
        raise ValueError("profile output is private: put it under logs/")
    if args.create_db == args.resume:
        raise ValueError("pass exactly one of --create-db (new run) or --resume")
    if args.create_db:
        if args.out.exists():
            raise RuntimeError(f"{args.out} exists; use --resume or a new run name")
        url = create_database(args.database)
    else:
        from sqlalchemy.engine import make_url
        from src.api.config import settings as live
        url = make_url(live.database_url).set(database=args.database).render_as_string(
            hide_password=False)
    args.out.mkdir(parents=True, exist_ok=True)
    # Settings and the engine bind DATABASE_URL at import, so the replay runs in a
    # fresh process that has only ever seen the scratch database.
    env = dict(os.environ, DATABASE_URL=url, ICE_TEST_DATABASE=args.database)
    argv = [a for a in sys.argv if a != "--create-db"] + ["--child"]
    os.execve(sys.executable, [sys.executable, *argv], env)


def run_profile(args) -> int:
    from contextlib import ExitStack
    from scripts.z1 import seed_v3
    slug = HISTORY_SLUG[args.history]
    original_load = seed_v3.load_plan
    clock = {}

    def load_plan(**kwargs):
        conversations, probes, quarantined, ineligible = original_load(**kwargs)
        clock["times"] = retime(conversations, slug, args.history)
        return conversations, probes, quarantined, ineligible

    argv = ["seed_v3.py", "--conversation", slug, "--limit", str(args.limit),
            "--probe-panel", "immediate", "--no-probe-controls",
            "--out", str(args.out / "seed.jsonl"), "--checkpoint-every", "10",
            "--worker-failure-policy", args.worker_failure_policy]
    if args.resume:
        argv.append("--resume")
    meta_path = args.out / "profile-meta.json"
    with (args.out / "profile.jsonl").open("a") as sink, ExitStack() as stack:
        profiler = Profiler(sink)
        install(profiler, stack)
        stack.enter_context(patch.object(seed_v3, "load_plan", load_plan))
        stack.enter_context(patch.object(sys, "argv", argv))
        sink.write(json.dumps({"stage": "run_start", "policy": POLICY, "argv": argv,
                               "resume": bool(args.resume), "wall": time.time()}) + "\n")
        sink.flush()
        status = seed_v3.main()
        if not meta_path.exists():
            meta_path.write_text(json.dumps(dict(
                policy=POLICY, history=args.history, slug=slug, limit=args.limit,
                clock_policy="experiments/z2/clock.py declared_times (provisional; D3 table not yet reviewed)",
                first_times=[t.isoformat() for t in clock.get("times", [])[:args.limit]]),
                indent=1) + "\n")
        return status


if __name__ == "__main__":
    raise SystemExit(main())
