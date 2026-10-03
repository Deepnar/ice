"""Actual v3 four-turn control for the manual seed crash and progress observer.

Disposable database only. Local models, original recorded replies, no cloud or
full campaign. This checks execution, not memory-quality gains.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.z1 import seed_v3
from scripts.z1.campaign_progress import ArtifactProgress, TerminalProgress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.out.name == "seed.jsonl"
    seed_v3.isolated_database()
    argv = ["seed_v3.py", "--conversation", "bb558b5f", "--limit", "4",
            "--probe-panel", "existing", "--checkpoint-every", "10", "--out", str(args.out)]
    reader = ArtifactProgress(args.out.parent, "seed", turns=4, probes=0)
    with TerminalProgress(reader, "1/5 seed local four-turn control"):
        with patch.object(sys, "argv", argv):
            assert seed_v3.main() == 0
    rows = [json.loads(line) for line in args.out.open()]
    assert [row["turn"] for row in rows if row["event"] == "written"] == [1, 2, 3, 4]
    assert [row["turn"] for row in rows if row["event"] == "clock"] == [1, 2, 3, 4]
    assert not any(row["event"].endswith("failed") for row in rows)
    assert {row["job"] for row in rows if row["event"] == "maintenance"} == set(seed_v3.MEMORY_JOBS)
    assert rows[-1]["complete_selected_corpus"] is False
    assert reader.observe().done == 4 and "durable 4" in reader.observe().detail
    # A limited fixture restores its final checkpoint and appends a resume
    # receipt. Only a complete full-corpus seed is byte-preserving on resume.
    before = [row for row in rows if row["event"] == "written"]
    with patch.object(sys, "argv", argv + ["--resume"]):
        assert seed_v3.main() == 0
    after = [json.loads(line) for line in args.out.open()]
    assert [row for row in after if row["event"] == "written"] == before
    assert reader.observe().done == 4 and "durable 4" in reader.observe().detail
    print("v3 actual four-turn seed/maintenance/progress/partial-resume control passed; no cloud calls or answer-quality claim")


if __name__ == "__main__":
    main()
