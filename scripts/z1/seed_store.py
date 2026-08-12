#!/usr/bin/env python3
"""Z1: populate the store from a real conversation, through the real pipeline.

**Why a seeding script and not just INSERTs.** The tuning question is "which
retrieval settings are best", and retrieval reads a store built by the rest of
ICE — the codex graph, chunks, clusters. Seeding bare `EpisodicMemory` rows
would leave the codex, chunk and cluster legs empty, so tuning their weights
would be tuning legs that have nothing to find. The user's question, and it was
the right one: *if we are not using the whole system, how are we even tuning it?*

So this runs the real components: the live classifier, the live embedder, and
(unless `--no-codex`) the real extractor against the real background model.

**⚑ THE COST IS PAID ONCE, ON PURPOSE.** The background model makes this
non-deterministic and slow, which is exactly what a tuning loop must not be. The
resolution is to populate once, `pg_dump` the result, and have every tuning run
restore that snapshot — so the LLM variance is spent here, at population, and
every measurement afterwards starts from byte-identical state. That also
disposes of **G38** (retrieval commits in three places, so probe N mutates the
store for probe N+1) without needing per-probe transaction surgery.

**⚑ TIMESTAMPS ARE SPREAD, NOT STAMPED "NOW".** If every turn shares a
timestamp, every memory is the same age, the recency bonus is constant, and any
recency knob measures as "cosmetic" — a verdict that would be an artifact of
this fixture rather than a fact about ICE. Turns are laid down across a
synthetic span ending `--end-days-ago` before now, in conversation order.

**Never truncates** (TRAPS #6, #15, #16). It owns its rows through the
conversation it creates and `--clean` removes exactly those.

Run:
  uv run python scripts/z1/seed_store.py --conv cca73c87 --limit 20 --no-codex
  uv run python scripts/z1/seed_store.py --conv cca73c87
  uv run python scripts/z1/seed_store.py --clean
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text  # noqa: E402

from scripts.z1.derive_retrieval_gt import load_conversations, turn_text  # noqa: E402
from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402
from src.memory.models import Conversation, EpisodicMemory  # noqa: E402

# Every episodic row this script writes carries this idempotency_key prefix, and
# --clean finds its conversations THROUGH those rows. Deliberately not a marker
# on `conversations` itself: that table has no free text column, and the one
# field that could carry a marker (`kind`) is indexed and filtered on by
# retrieval, so borrowing it would change what the fixture retrieves — a fixture
# that alters the thing it measures.
MARK = "z1seed"
MANIFEST = "experiments/curation_files/seeded_store.json"


def clean(db) -> int:
    """Delete only what this script created — episodic AND codex.

    ⚠ The first version cleaned episodic rows only and left **307 entities and
    380 edges** behind from a 10-turn timing run. That is the TRAPS #6 residue
    pattern, and codex residue is worse than episodic residue for this
    instrument: leftover entities and edges are exactly what the codex leg
    retrieves, so a later scoring run would be measuring a store nobody
    described.

    Edges are traceable — `source_batch` carries the batch id of the turn that
    produced them. **Entities are not**: `CodexEntity` has no source column
    because `get_or_create_entity` dedupes globally by canonical name. So the
    manifest records the entity ids that existed BEFORE seeding, and anything
    outside that set was created by it. Exact, and it degrades safely: with no
    manifest the entity sweep is skipped rather than guessed at.
    """
    ids = [r[0] for r in db.execute(
        text("SELECT DISTINCT conversation_id FROM episodic_memory "
             "WHERE idempotency_key LIKE :m"), {"m": f"{MARK}-%"}).all()]
    batch_ids = [r[0] for r in db.execute(
        text("SELECT batch_id FROM episodic_memory WHERE idempotency_key LIKE :m"),
        {"m": f"{MARK}-%"}).all()]

    # Each statement stands alone: a failure here must not abort the rest of the
    # sweep. The first version let one bad column name (`codex_events.batch_id`,
    # which does not exist — it is `batch_source`) roll back and abandon the
    # episodic deletes, leaving a half-cleaned store, which is worse than either
    # extreme.
    if batch_ids:
        for sql in (
            "DELETE FROM codex_events WHERE batch_source = ANY(:b)",
            "DELETE FROM codex_edges WHERE source_batch = ANY(:b)",
            "DELETE FROM codex_relation_gaps WHERE batch_id = ANY(:b)",
            # procedural_memory has NO foreign key to conversations or
            # episodic_memory, so the schema walk below cannot find it — it
            # links through a `source_batch_ids` ARRAY instead. Left behind on
            # the first fixed cleanup, and it feeds a retrieval leg, so residue
            # here is residue the scorer would read.
            "DELETE FROM procedural_memory WHERE source_batch_ids && CAST(:b AS uuid[])",
        ):
            try:
                db.execute(text(sql), {"b": batch_ids})
                db.commit()
            except Exception as exc:
                print(f"  ! {sql.split()[2]}: {exc}")
                db.rollback()

    if os.path.exists(MANIFEST):
        try:
            pre = json.load(open(MANIFEST)).get("pre_existing_entity_ids")
        except Exception:
            pre = None
        if pre is None:
            print("  ! manifest has no pre_existing_entity_ids — entities left alone")
        else:
            try:
                # An EMPTY pre-list is the common case (a clean store before
                # seeding) and must delete everything, so the cast is explicit:
                # an untyped empty array makes `= ANY()` fail type inference.
                n = db.execute(text(
                    "DELETE FROM codex_entities "
                    "WHERE NOT (id = ANY(CAST(:pre AS uuid[])))"),
                    {"pre": pre or []}).rowcount
                db.commit()
                print(f"  removed {n} codex entities created by seeding")
            except Exception as exc:
                print(f"  ! codex_entities: {exc}")
                db.rollback()

    if not ids:
        print("nothing to clean")
        return 0

    # ⚑ THE DEPENDENT TABLES ARE DISCOVERED FROM THE LIVE SCHEMA, not listed.
    # A hardcoded list is a list that goes stale: this cleanup already broke
    # once when clustering joined the pipeline and `context_clusters` started
    # referencing the conversation, and the resulting foreign-key violation
    # aborted the whole delete and left a half-cleaned store. Asking the
    # database which tables point at ours cannot fall out of date.
    def dependents(target: str) -> list[tuple[str, str]]:
        rows = db.execute(text("""
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND ccu.table_name = :t AND tc.table_name <> :t
        """), {"t": target}).all()
        return [(r[0], r[1]) for r in rows]

    removed = Counter()
    for tbl, col in dependents("episodic_memory"):
        try:
            n = db.execute(text(
                f"DELETE FROM {tbl} WHERE {col} IN "
                f"(SELECT id FROM episodic_memory WHERE conversation_id = ANY(:i))"),
                {"i": ids}).rowcount
            db.commit()
            removed[tbl] += n
        except Exception as exc:
            print(f"  ! {tbl}: {str(exc)[:120]}")
            db.rollback()

    for tbl, col in dependents("conversations"):
        if tbl == "episodic_memory":
            continue                       # deleted below, after its own dependents
        try:
            n = db.execute(text(f"DELETE FROM {tbl} WHERE {col} = ANY(:i)"),
                           {"i": ids}).rowcount
            db.commit()
            removed[tbl] += n
        except Exception as exc:
            print(f"  ! {tbl}: {str(exc)[:120]}")
            db.rollback()

    for sql in ("DELETE FROM episodic_memory WHERE conversation_id = ANY(:i)",
                "DELETE FROM conversations WHERE id = ANY(:i)"):
        try:
            removed[sql.split()[2]] += db.execute(text(sql), {"i": ids}).rowcount
            db.commit()
        except Exception as exc:
            print(f"  ! {sql.split()[2]}: {str(exc)[:160]}")
            db.rollback()

    detail = ", ".join(f"{k} {v}" for k, v in sorted(removed.items()) if v)
    print(f"removed {len(ids)} seeded conversation(s): {detail or 'nothing'}")
    return len(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default=None, help="8-char id; default all three")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-codex", action="store_true",
                    help="skip extraction — fast, but leaves the codex leg empty")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--span-days", type=int, default=120,
                    help="lay the turns down across this many days")
    ap.add_argument("--end-days-ago", type=int, default=2)
    args = ap.parse_args()

    db = SessionLocal()
    if args.clean:
        clean(db)
        db.close()
        return 0

    convs = load_conversations()
    if args.conv:
        convs = {k: v for k, v in convs.items() if k == args.conv}
    if not convs:
        print("no conversations loaded")
        return 1

    from src.classifier.classifier import PyTorchClassifier
    from src.memory.embedder import get_embedder

    embedder = get_embedder()
    clf = PyTorchClassifier(model_path=settings.classifier_model_path,
                            schema_path=settings.label_schema_path)
    stats = Counter()
    # Snapshot what exists BEFORE we write anything — the only way to tell a
    # seeded entity from a pre-existing one, since CodexEntity has no source
    # column (see clean()).
    pre_ids = [r[0] for r in db.execute(text("SELECT id FROM codex_entities")).all()]
    manifest = {"marker": MARK, "conversations": {},
                "pre_existing_entity_ids": [str(i) for i in pre_ids]}
    print(f"pre-existing codex entities: {len(pre_ids)}")

    from src.workers.bg_client_factory import get_bg_model_name
    from src.workers.post_flight import evaluate_turn
    # ⚠ model_used is not a label — procedural_extractor CALLS it. Passing
    # a tag ("z1-seed") 404s the whole post-flight chain.
    seed_model = get_bg_model_name()

    for cid, meta in sorted(convs.items()):
        turns = meta["turns"][:args.limit] if args.limit else meta["turns"]

        # RESUME: reuse this conversation's existing shell if a previous run
        # created one, so re-running tops up instead of duplicating. Found via
        # the deterministic idempotency key below — the old key carried a random
        # suffix, which made resume impossible because no key could be predicted.
        existing = db.execute(text(
            "SELECT DISTINCT conversation_id FROM episodic_memory "
            "WHERE idempotency_key LIKE :m"), {"m": f"{MARK}-{cid}-%"}).first()
        if existing:
            conv_id = existing[0]
            print(f"── {cid}: resuming into {conv_id}")
        else:
            conv = Conversation(memory_scope_type="auto")
            db.add(conv)
            db.commit()
            conv_id = conv.id
            print(f"── {cid}: seeding {len(turns)} turns into {conv_id}")

        class _C:                      # tiny shim so the loop below reads the same
            id = conv_id
        conv = _C()

        # Oldest first, ending `end_days_ago` before now — see the module note on
        # why a single shared timestamp would silently neuter every recency knob.
        end = datetime.now(timezone.utc) - timedelta(days=args.end_days_ago)
        step = timedelta(days=args.span_days) / max(1, len(turns))
        turn_ids = []

        for i, t in enumerate(turns):
            user = (t.get("user_input") or "").strip()
            assistant = (t.get("ai_response") or "").strip()
            if not user and not assistant:
                continue
            tn = t.get("turn_number")
            # ⚑ DETERMINISTIC, so the turn can be recognised on a later run.
            # The first version appended a random hex suffix, which made resume
            # impossible: nothing could predict the key to look for.
            ikey = f"{MARK}-{cid}-{tn}"

            row = db.query(EpisodicMemory).filter_by(idempotency_key=ikey).first()
            if row is None:
                ts = end - step * (len(turns) - 1 - i)
                try:
                    c = clf.classify(user[:2000])
                except Exception as exc:
                    print(f"  ! classify turn {tn}: {exc}")
                    stats["classify_failed"] += 1
                    continue
                row = EpisodicMemory(
                    conversation_id=conv.id,
                    batch_id=uuid.uuid4(),
                    timestamp=ts,
                    topic_tags=list(c.topic_tags or []),
                    intent_tags=list(c.intent_tags or []),
                    context_reliance=c.context_reliance,
                    raw_text=f"User: {user}\n\nAssistant: {assistant}",
                    lossless_flag=True,
                    embedding=embedder.encode(user or assistant,
                                              convert_to_tensor=False).tolist(),
                    idempotency_key=ikey,
                )
                db.add(row)
                db.commit()          # per TURN, not per conversation: an interrupt
                stats["turns"] += 1  # now costs one turn, not an hour of extraction
            else:
                stats["turns_already_present"] += 1

            turn_ids.append({"turn_number": tn, "episodic_id": str(row.id),
                             "batch_id": str(row.batch_id)})

            if not args.no_codex:
                # ⚑ THE REAL POST-FLIGHT, not a hand-rolled extraction call.
                # evaluate_turn runs density qualification, the grounded summary,
                # chunking, codex extraction AND procedural extraction — the four
                # stages the previous version skipped, which would have left the
                # chunk, procedural and cluster legs empty and made their weights
                # untunable. Its own docstring guarantees the chain is
                # self-idempotent ("a retry after a partial chain failure
                # completes the missing stages"), which is what makes this script
                # resumable rather than restart-only.
                try:
                    evaluate_turn(batch_id=str(row.batch_id), prompt=user,
                                  response=assistant,
                                  conversation_id=str(conv.id),
                                  model_used=seed_model)
                    stats["post_flight"] += 1
                except Exception as exc:
                    print(f"    ! post_flight turn {tn}: {type(exc).__name__}: {exc}")
                    stats["post_flight_failed"] += 1

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(turns)}  (post-flight {stats['post_flight']})")

        manifest["conversations"][cid] = {"conversation_id": str(conv.id),
                                          "turns": turn_ids}
        with open(MANIFEST, "w") as fh:      # written per conversation, so an
            json.dump(manifest, fh, indent=2)  # interrupt still leaves a usable map

    # ── Clustering: a periodic job in production, run here as a catch-up pass
    #    over everything just seeded. Without it `_relevant_cluster_ids` has
    #    nothing to match and cluster-scoped retrieval is silently inert.
    if not args.no_codex:
        try:
            from src.workers.clustering import run_cluster_assignment, run_cluster_merge
            conv_ids = [m["conversation_id"] for m in manifest["conversations"].values()]
            print("\nclustering …")
            print(f"  assignment: {run_cluster_assignment(db, conversation_ids=conv_ids)}")
            print(f"  merge:      {run_cluster_merge(db, conversation_ids=conv_ids)}")
        except Exception as exc:
            print(f"  ! clustering: {type(exc).__name__}: {exc}")
            db.rollback()

    counts = {t: db.execute(text(f"SELECT count(*) FROM {t}")).scalar()
              for t in ("episodic_memory", "codex_entities", "codex_edges",
                        "episodic_chunks", "procedural_memory", "context_clusters",
                        "batch_summaries", "codex_relation_gaps")}
    print(f"\nseeded {stats['turns']} new turns "
          f"({stats['turns_already_present']} already present) · "
          f"post-flight ok {stats['post_flight']}, failed {stats['post_flight_failed']} "
          f"· classify failures {stats['classify_failed']}")
    print(f"store now: {counts}")
    print(f"manifest → {MANIFEST}  (turn_number → episodic id, for scoring)")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
