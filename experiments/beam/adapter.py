"""FINAL's BEAM adapter (FINAL_experiments.md §2.1): a BEAM conversation to Z2-format
originals and asks, and BEAM-style nugget scoring.

Run natively: replay the whole conversation, ask its 20 probing questions once after the
last turn. Timestamps come from BEAM's own session anchors (one per session, e.g.
"March-15-2024"), turns spaced TURN_GAP within a session — BEAM has real session dates, so
Z2's declared clock is not imposed. The generator's bookkeeping markers ("->-> 3,8") are
kept: BEAM's own baselines saw them (decided 2026-10-26).

Scoring follows BEAM's protocol in our own words, labelled non-official: each rubric nugget
is scored 0 / 0.5 / 1 by the judge and averaged; event ordering adds Kendall tau-b between
the rubric order and the order the answer gives.
"""

import ast
import json
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
TURN_GAP, SESSION_START = timedelta(minutes=3), "19:00"

NUGGET_SYSTEM = """You grade one ANSWER to a QUESTION about a long conversation against NUGGETS, the
atomic points a correct answer must contain (from the benchmark's reference). For each nugget
give 1 if the answer states it, 0.5 if it states it partly or vaguely, 0 if it is missing or
contradicted. For an abstention question the single nugget is that the answer declines for lack
of information. Return only JSON: {"scores": [one number per nugget, in order], "reason": "..."}"""

ORDER_SYSTEM = """ITEMS are events from a conversation, in their true order. ANSWER lists events in some
order. For each item give the position (1-based) at which the answer places the matching event,
or null if the answer does not mention it. Return only JSON: {"positions": [...]}"""


def rows(tier):
    import pyarrow.parquet as pq
    return pq.read_table(HERE / "data" / f"{tier}.parquet").to_pylist()


def anchor_date(text):
    return datetime.strptime(text, "%B-%d-%Y").date()


def to_originals(row):
    """[(turn_number, timestamp, prompt, response, provenance)] and message id -> turn."""
    out, id_to_turn, turn = [], {}, 0
    for session in row["chat"]:
        anchors = [m["time_anchor"] for m in session if m["time_anchor"] not in (None, "None")]
        if not anchors:
            raise ValueError(f"{row['conversation_id']}: a session without a time anchor")
        start = datetime.fromisoformat(f"{anchor_date(anchors[0]).isoformat()}T{SESSION_START}")
        users = [m for m in session if m["role"] == "user"]
        pending = None
        for m in session:
            if m["role"] == "user":
                pending = m
            elif pending is not None:
                turn += 1
                k = users.index(pending)
                id_to_turn[int(pending["id"])] = id_to_turn[int(m["id"])] = turn
                out.append(dict(turn_number=turn, timestamp=(start + k * TURN_GAP).isoformat() + "Z",
                                timestamp_provenance="beam_time_anchor",
                                prompt=pending["content"], response=m["content"]))
                pending = None
    return out, id_to_turn


def source_turns(question, id_to_turn):
    raw = question.get("source_chat_ids")
    if not raw:
        return []
    value = ast.literal_eval(raw) if isinstance(raw, str) else raw
    ids = flat(list(value.values()) if isinstance(value, dict) else value)
    return sorted({id_to_turn[int(i)] for i in ids if int(i) in id_to_turn})


def flat(value):
    """Message ids at any depth: an event_ordering question groups one event's messages in a nested
    list ([[24, 26, 28], [146], ...] — 5 of the chosen conversations' questions, 2026-11-01)."""
    if isinstance(value, (list, tuple)):
        return [i for v in value for i in flat(v)]
    return [value]


def to_asks(row, id_to_turn, last_turn):
    asks = []
    for qtype, items in ast.literal_eval(row["probing_questions"]).items():
        for i, q in enumerate(items):
            rubric = q.get("rubric")
            nuggets = ast.literal_eval(rubric) if isinstance(rubric, str) else (rubric or [])
            asks.append(dict(id=f"BEAM:{row['conversation_id']}:{qtype}:{i}", kind="question", type=qtype,
                             question=q["question"], checkpoint_turn=last_turn,
                             reference=dict(nuggets=nuggets, answer=q.get("answer") or q.get("ideal_response")
                                            or q.get("ideal_answer") or q.get("ideal_summary"),
                                            source_turns=source_turns(q, id_to_turn)),
                             expected_abstention=qtype == "abstention", difficulty=q.get("difficulty")))
    return asks


def kendall_tau_b(positions):
    """tau-b between the true order 1..n and the answer's positions (missing items dropped)."""
    pairs = [(i, p) for i, p in enumerate(positions) if p is not None]
    n = len(pairs)
    if n < 2:
        return 0.0
    conc = disc = tie_x = tie_y = 0
    for a in range(n):
        for b in range(a + 1, n):
            dx, dy = pairs[a][0] - pairs[b][0], pairs[a][1] - pairs[b][1]
            if dy == 0:
                tie_y += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    total = n * (n - 1) / 2
    denom = ((total - tie_x) * (total - tie_y)) ** 0.5
    return (conc - disc) / denom if denom else 0.0


def score(ask, answer, judge):
    """judge(system, prompt) -> dict. Nugget mean; event ordering also gets tau-b."""
    nuggets = ask["reference"]["nuggets"]
    got = judge(NUGGET_SYSTEM, json.dumps(dict(QUESTION=ask["question"], ANSWER=answer, NUGGETS=nuggets),
                                          ensure_ascii=False))
    scores = [min(1.0, max(0.0, float(s))) for s in (got or {}).get("scores") or []]
    if len(scores) != len(nuggets):
        raise ValueError(f"{ask['id']}: {len(scores)} scores for {len(nuggets)} nuggets")
    out = dict(nugget_mean=sum(scores) / len(scores) if scores else 0.0, nuggets=scores)
    if ask["type"] == "event_ordering":
        pos = judge(ORDER_SYSTEM, json.dumps(dict(ITEMS=nuggets, ANSWER=answer), ensure_ascii=False))
        out["kendall_tau_b"] = kendall_tau_b((pos or {}).get("positions") or [])
    return out


def export(out_dir):
    """Originals + asks for every conversation in MANIFEST.json."""
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    out_dir = Path(out_dir)
    for tier, t in manifest["tiers"].items():
        chosen = {c["conversation_id"] for c in t["conversations"]}
        for row in rows(tier):
            if row["conversation_id"] not in chosen:
                continue
            originals, id_to_turn = to_originals(row)
            asks = to_asks(row, id_to_turn, originals[-1]["turn_number"])
            d = out_dir / tier / row["conversation_id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "originals.json").write_text(json.dumps(originals, ensure_ascii=False) + "\n")
            (d / "asks.json").write_text(json.dumps(asks, indent=1, ensure_ascii=False) + "\n")
    return out_dir


if __name__ == "__main__":
    import sys
    print(export(sys.argv[1] if len(sys.argv) > 1 else "logs/final-beam"))
