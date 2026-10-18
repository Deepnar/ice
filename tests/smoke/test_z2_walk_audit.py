"""Z2 D4 forward-walk audit: intervals, candidate turns inside the interval only,
and the review queue on walk/ledger disagreement."""

import json
from types import SimpleNamespace as NS

import numpy as np

from experiments.z2 import walk_audit as W


def test_interval_runs_from_the_latest_citation_to_the_ask():
    o = dict(checkpoint_turn=80, reference=dict(citations=[dict(turn=12), dict(turn=40)]))
    assert W.interval(o) == (40, 80)


def test_candidates_come_only_from_inside_the_interval():
    rows = [(t, "user", f"m{t}") for t in range(1, 11)]
    vectors = np.eye(10)
    shown = W.candidates((rows, vectors), np.ones(10), 3, 6, top_k=10)
    assert [r[0] for r in shown] == [4, 5, 6]


def test_disagreements_go_to_the_queue(tmp_path, monkeypatch):
    originals = [dict(turn_number=t, prompt=f"the cat is now called Tofu {t}", response="ok") for t in range(1, 31)]
    rows = [(t, "user", originals[t - 1]["prompt"]) for t in range(1, 31)]
    index = (rows, np.random.default_rng(0).random((30, 4)))
    monkeypatch.setattr(W.Q, "embed", lambda texts, device="cpu": np.ones((len(texts), 4)))
    fams = [dict(id="f1", type="update", entries=[]), dict(id="f2", type="fact_recall", entries=[])]
    occ = [dict(family=f["id"], type=f["type"], checkpoint_turn=30, question=f"what is the cat called ({f['id']})",
                reference=dict(required=["Miso"], citations=[dict(turn=5)])) for f in fams]
    monkeypatch.setattr(W, "ledger_changed", lambda fam, *_: fam["id"] == "f2")
    reply = {"changes": True, "turn": 20, "role": "user", "quote": "the cat is now called Tofu", "new_answer": "Tofu"}
    replies = [reply, {"changes": False}]
    gen = lambda messages, **_: NS(text=json.dumps(replies.pop(0)), usage={})
    (tmp_path / "C").mkdir()
    summary, queue = W.audit("C", fams, occ, {}, originals, index, tmp_path, models={"muse": (gen, {})})
    assert summary["walk_only"] == 1 and summary["ledger_only"] == 1 and summary["agree"] == 0
    assert {q["kind"] for q in queue} == {"walk_change_not_in_ledger", "ledger_change_walk_missed"}
    walk_row = next(q for q in queue if q["kind"] == "walk_change_not_in_ledger")
    assert walk_row["quote_ok"] is True
