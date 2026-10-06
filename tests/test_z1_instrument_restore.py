"""Two reviewed instrument boundaries with actual 35-table snapshot restores.

Synthetic code identities stand in for review pairs; isolation tests separately
require exact Git/file registry hashes. No original experiment is restored.
"""
import argparse
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.z1 import replay_checkpoint as recovery, seed_v3, snapshot
from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seed_v3.isolated_database()
    output = seed_v3.private_output(str(args.out))
    if output.exists() or output.with_suffix(".recovery").exists():
        raise ValueError("use a fresh synthetic fixture output")
    identity = {"code_sha256": "synthetic-0", "settings_sha256": "frozen", "models": "frozen"}
    changes = {"scripts/z1/replay_checkpoint.py": {"before": "fixture-a", "after": "fixture-b"}}
    current = ["synthetic-1"]
    baselines = {"fixture-first": "synthetic-0", "fixture-second": "synthetic-1"}
    with SessionLocal() as db:
        conversation = Conversation()
        db.add(conversation)
        db.flush()
        row = EpisodicMemory(conversation_id=conversation.id, batch_id=uuid.uuid4(),
                             idempotency_key="synthetic-restored-source", raw_text="Original one",
                             context_reliance="Long_Term_Memory")
        db.add(row)
        db.commit()
        row_id = row.id
        with (patch.object(recovery, "code_digest", lambda ref=None: baselines[ref] if ref else current[0]),
              patch.object(recovery, "instrument_changes", lambda _: changes),
              patch.object(recovery, "registered_repair", lambda *_: {"reason": "synthetic reviewed boundary"})):
            point = recovery.Checkpoints(output.with_suffix(".recovery"), output, identity)
            try:
                with output.open("w") as sink:
                    sink.write('{"event":"written","turn":1}\n')
                    point.capture(sink, {"completed": {"synthetic": 1}})
                first_prefix = output.read_bytes()
                first = point.prepare_instrument_continuation("fixture-first")
                point.receipt_path(first["from_identity"], first["to_identity"]).rename(
                    point.root / "instrument-continuation.json")
                legacy_receipt = (point.root / "instrument-continuation.json").read_bytes()
                original_fingerprint = snapshot.fingerprints()
                row.raw_text = "Unfinished change"
                db.commit()
                with output.open("a") as sink:
                    sink.write('{"event":"failed"}\n')
                point.identity = first["to_identity"]
                assert point.recover()["completed"] == {"synthetic": 1}
                assert snapshot.fingerprints() == original_fingerprint
                db.expire_all()
                db.get(EpisodicMemory, row_id).raw_text = "Original two"
                db.commit()
                with output.open("a") as sink:
                    sink.write('{"event":"written","turn":2}\n')
                    point.capture(sink, {"completed": {"synthetic": 2}})
                second_prefix = output.read_bytes()
                pointer = point.pointer.read_bytes()
                fingerprint = snapshot.fingerprints()
                current[0] = "synthetic-2"
                second = point.prepare_instrument_continuation("fixture-second")
                assert point.pointer.read_bytes() == pointer and output.read_bytes() == second_prefix
                assert snapshot.fingerprints() == fingerprint
                assert (point.root / "instrument-continuation.json").read_bytes() == legacy_receipt
                db.get(EpisodicMemory, row_id).raw_text = "Another unfinished change"
                db.commit()
                with output.open("a") as sink:
                    sink.write('{"event":"failed"}\n')
                point.identity = second["to_identity"]
                assert point.recover()["completed"] == {"synthetic": 2}
                assert snapshot.fingerprints() == fingerprint
                markers = [r for r in map(json.loads, output.read_text().splitlines())
                           if r["event"] == "instrument_continuation"]
                assert [r["to_identity"]["code_sha256"] for r in markers] == ["synthetic-1", "synthetic-2"]
                assert (point.root / first["archive"] / "trace-prefix.jsonl").read_bytes() == first_prefix
                assert (point.root / second["archive"] / "trace-prefix.jsonl").read_bytes() == second_prefix
                assert len(list(point.root.glob("unfinished-*.jsonl"))) == 2
            finally:
                point.close()
    print("v3 synthetic two-boundary recovery: all35-table row fingerprints restored, both receipts/prefixes retained, "
          "unfinished tails archived; no campaign or cloud calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
