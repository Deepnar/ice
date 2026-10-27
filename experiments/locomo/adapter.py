"""FINAL's LoCoMo adapter (pre-registration r2 §4: a small, clearly labelled extra table).

LoCoMo — snap-research/locomo at commit 3eb6f2c (data/locomo10.json, sha256 79fa87e9…, pinned in
MANIFEST.json; the repository states no licence, which the paper says): 10 conversations between
two named people, 19–32 dated sessions each, 1,986 questions asked after the last session.

To Z2-format originals: the conversation's first speaker (speaker_a) is always on ICE's user
side and the second (speaker_b) always on its assistant side — never swapped between sessions
(124 of LoCoMo's 272 sessions open with speaker_b; pairing by position would hand that person
the user side there, the role mix other LoCoMo setups were criticised for). A pair is a
speaker_a message and the speaker_b reply after it; a session opened by speaker_b starts with an
empty user side, one closed by speaker_a ends with an empty assistant side. Each text keeps its
speaker's name; a shared photo becomes "[shares a photo: <caption>]". Turns are timed from the
session's own date, TURN_GAP apart, so the replay runs on LoCoMo's clock (provenance
"locomo_session_date", a source clock — experiments/z2/clock.py).

Questions: every QA item, asked once after the last turn (checkpoint = the last turn). The
reference is its answer, its citations the turns of its evidence ids ("D<session>:<n>"; an id
naming no message is counted and dropped). Category 5 (adversarial — the conversation never says
it) is a never-said question: expected abstention, reported apart; categories 1–4 are the table
comparable with other papers. Each question is its own family. The ledger is empty: an answer's
unsupported claim is checked against the original messages only.

Native score: token F1 against the answer (LoCoMo's own metric; its per-category variants are not
reproduced), beside our judge's M1/M2.

    uv run --no-sync python experiments/locomo/adapter.py export [out-dir]        # logs/final-locomo
"""

import collections
import hashlib
import json
import re
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "locomo10.json"
MANIFEST = HERE / "MANIFEST.json"
TURN_GAP = timedelta(minutes=3)
PROVENANCE = "locomo_session_date"
# LoCoMo's category numbers, as memory papers commonly read them (the dataset itself only numbers them)
CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


def load(path=DATA):
    raw = Path(path).read_bytes()
    pinned = json.loads(MANIFEST.read_text())["sha256"]
    if hashlib.sha256(raw).hexdigest() != pinned:
        raise ValueError(f"{path}: not the pinned LoCoMo file (sha256 {pinned[:12]}…)")
    return json.loads(raw)


def session_date(text):
    """'1:56 pm on 8 May, 2023' -> datetime."""
    return datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y")


def message_text(m):
    text = m.get("text") or ""
    if m.get("blip_caption"):
        text = f"{text} [shares a photo: {m['blip_caption']}]".strip()
    return f"{m['speaker']}: {text}"


def sessions(conv):
    numbers = sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
    return [(n, session_date(conv[f"session_{n}_date_time"]), conv[f"session_{n}"]) for n in numbers]


def pairs(messages, user):
    """(user-side message or None, assistant-side message or None) in order; *user* is speaker_a."""
    out = []
    for m in messages:
        if m["speaker"] == user or not out or out[-1][1] is not None:
            out.append([m, None] if m["speaker"] == user else [None, m])
        else:
            out[-1][1] = m
    return out


def to_originals(sample):
    """[{turn_number, timestamp, timestamp_provenance, prompt, response}] and dia_id -> turn."""
    out, dia_to_turn, turn = [], {}, 0
    user = sample["conversation"]["speaker_a"]
    for _, start, messages in sessions(sample["conversation"]):
        for k, (first, second) in enumerate(pairs(messages, user)):
            turn += 1
            for m in (first, second):
                if m is not None:
                    dia_to_turn[m["dia_id"]] = turn
            out.append(dict(turn_number=turn, timestamp=(start + k * TURN_GAP).isoformat() + "Z",
                            timestamp_provenance=PROVENANCE, prompt=message_text(first) if first else "",
                            response=message_text(second) if second else ""))
    return out, dia_to_turn


def key(sample_id):
    return f"LOCOMO-{sample_id}"


def questions(sample, dia_to_turn, last_turn):
    """(asks for the read passes, occurrences for grade.py, dropped evidence ids)."""
    asks, occurrences, dropped = [], [], 0
    for i, q in enumerate(sample["qa"]):
        family = f"{key(sample['sample_id'])}:Q{i:03d}"
        ask_id = f"{family}@{last_turn}#0"
        never_said = q["category"] == 5
        evidence = [e.strip() for e in q.get("evidence") or [] for e in str(e).split(";")]
        turns = sorted({dia_to_turn[e] for e in evidence if e in dia_to_turn})
        dropped += sum(1 for e in evidence if e not in dia_to_turn)
        qtype = CATEGORIES.get(q["category"], f"category_{q['category']}")
        asks.append(dict(id=ask_id, kind="question", type=qtype, question=q["question"], checkpoint_turn=last_turn))
        occurrences.append(dict(family=family, type=qtype, checkpoint_turn=last_turn, repeat_index=0,
                                question=q["question"], expected_abstention=never_said, category=q["category"],
                                reference=dict(required=[] if never_said else [str(q["answer"])], superseded=[],
                                               citations=[dict(turn=t) for t in turns], answer=q.get("answer"))))
    return asks, occurrences, dropped


def export(out_dir="logs/final-locomo", data=None):
    out_dir, report = Path(out_dir), {}
    for sample in data if data is not None else load():
        originals, dia_to_turn = to_originals(sample)
        asks, occurrences, dropped = questions(sample, dia_to_turn, originals[-1]["turn_number"])
        d = out_dir / sample["sample_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "originals.json").write_text(json.dumps(originals, ensure_ascii=False) + "\n")
        (d / "asks.json").write_text(json.dumps(asks, indent=1, ensure_ascii=False) + "\n")
        (d / "occurrences.json").write_text(json.dumps(occurrences, indent=1, ensure_ascii=False) + "\n")
        (d / "ledger.json").write_text(json.dumps(dict(entries=[]), indent=1) + "\n")
        report[sample["sample_id"]] = dict(turns=len(originals), questions=len(asks), evidence_ids_dropped=dropped)
    (out_dir / "export.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


# ── LoCoMo's native score ────────────────────────────────────────────────────

def normalize(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, truth):
    p, t = normalize(prediction).split(), normalize(truth).split()
    common = sum((collections.Counter(p) & collections.Counter(t)).values())
    if not p or not t or not common:
        return 0.0
    precision, recall = common / len(p), common / len(t)
    return 2 * precision * recall / (precision + recall)


def native(units_dir, answers_dir, occurrences_path):
    """Per condition and category: mean token F1 of the answers (categories 1–4; category 5 has
    no answer to overlap with and is scored by the judge's abstention only)."""
    occ = {f"{o['family']}@{o['checkpoint_turn']}#{o['repeat_index']}": o
           for o in json.loads(Path(occurrences_path).read_text())}
    index = json.loads((Path(answers_dir) / "index.json").read_text())["index"]
    rows = {}
    for line in (Path(answers_dir) / "answers.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["prompt_sha256"]] = r
    cells = collections.defaultdict(list)
    for k, sha in index.items():
        cond, ask = k.split("|", 1)
        o, r = occ.get(ask), rows.get(sha)
        if o is None or r is None or r.get("finish") != "complete" or o["expected_abstention"]:
            continue
        cells[(cond, o["type"])].append(token_f1(r["text"], o["reference"]["answer"]))
    out = collections.defaultdict(dict)
    for (cond, qtype), scores in sorted(cells.items()):
        out[cond][qtype] = dict(n=len(scores), token_f1=round(sum(scores) / len(scores), 4))
    return dict(out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        print(json.dumps(export(*sys.argv[2:3]), indent=1))
    else:
        raise SystemExit(__doc__)
