"""R10's other side (experiments/z2/nomem_cost.py): only the maintainer's no-memory messages are
sampled, the reads never write the source store, and the report counts each bar from one read."""
import json
import re
import sys

import pytest

from experiments.z2 import nomem_cost as N


def test_only_personal_no_memory_messages_are_candidates(tmp_path):
    rows = [dict(id="p1", source="personal", prompt="hi there", label=dict(context_reliance="Zero_Shot")),
            dict(id="p2", source="personal", prompt="what did I say", label=dict(context_reliance="Long_Term_Memory")),
            dict(id="l1", source="lmsys", prompt="hello", label=dict(context_reliance="Zero_Shot")),
            dict(id="p3", source="personal", prompt="  ", label=dict(context_reliance="Zero_Shot"))]
    path = tmp_path / "labeled.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert [r["id"] for r in N.candidates(path)] == ["p1"]


def test_the_reads_never_run_on_the_source_store():
    with pytest.raises(ValueError, match="copy"):
        N.clone("ice_z2_main_b", "ice_z2_main_b")
    with pytest.raises(ValueError, match="Z2 database"):
        N.clone("ice_z2_main_b", "ice_db")


def test_the_report_counts_each_bar_beside_r10(tmp_path, capsys):
    out, r10 = tmp_path / "out", tmp_path / "r10"
    (out / "units").mkdir(parents=True)
    r10.mkdir()
    (out / "sample.json").write_text(json.dumps(dict(seed=0, left_out_stored=2, messages=[])))
    for i, (b2, best) in enumerate([(False, -3.0), (False, 0.5), (False, None), (True, None)]):
        (out / "units" / f"m{i}.json").write_text(json.dumps(dict(
            id=f"m{i}", gate=dict(b2_retrieve=b2, rescue_best_score=best), selected_tokens=100, seconds=5)))
    for i, best in enumerate([4.0, -1.5]):
        (r10 / f"q{i}.json").write_text(json.dumps(dict(stage=dict(gate=dict(rescue_best_score=best)))))
    N.report(out, r10_units=r10)
    text = capsys.readouterr().out
    assert "fetched memory on 1" in text and "said no on 3" in text and "found nothing) on 1" in text
    bar0 = next(line for line in text.splitlines() if line.strip().startswith("+0"))
    assert re.findall(r"(\d+) \(", bar0) == ["1", "1"]             # needed: 4.0 only; not needed: 0.5 only


def test_output_outside_logs_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["nomem_cost.py", "report", "--out", "src/x"])
    with pytest.raises(ValueError, match="private"):
        N.main()
