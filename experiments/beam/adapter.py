"""FINAL's BEAM adapter (FINAL_experiments.md §2.1): a BEAM conversation to Z2-format
originals and asks, and BEAM's own scoring.

Run natively: replay the whole conversation, ask its 20 probing questions once after the
last turn. Timestamps come from BEAM's own session anchors (one per session, e.g.
"March-15-2024"), turns spaced TURN_GAP within a session — BEAM has real session dates, so
Z2's declared clock is not imposed. The generator's bookkeeping markers ("->-> 3,8") are
kept: BEAM's own baselines saw them (decided 2026-10-26).

Scoring is BEAM's own (its repository's prompts and metrics, verbatim; the judge is FINAL's, so
labelled non-official): each rubric nugget scored 0 / 0.5 / 1 in its own call and averaged; event
ordering scored, as BEAM reports it, by normalised Kendall tau-b between the rubric's order and
the answer's lines aligned to it.
"""

import ast
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
TURN_GAP, SESSION_START = timedelta(minutes=3), "19:00"

# BEAM's own judge, verbatim (github.com/mohammadtavakoli78/BEAM @ b2da22e, MIT): src/prompts.py
# unified_llm_judge_base_prompt, one rubric nugget per call, and compute_metrics.llm_equivalence's
# messages for aligning an event-ordering answer (Z2 F27: the adapter had paraphrased them).
UNIFIED_JUDGE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT
A compliant response must be **on-topic** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
   
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""

EQUIVALENCE_SYSTEM = """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """

EQUIVALENCE_USER = """First snippet: {first} \n
                       Second snippet: {second}
                    """


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


def parse_json_response(response):
    """BEAM's compute_metrics.parse_json_response: a fenced or bare JSON object (ValueError when none;
    BEAM then repairs the text, this adapter leaves the nugget for the next run)."""
    response = response.strip()
    if response.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    match = re.search(r'(\{.*?\}|\[.*?\])', response, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("No valid JSON found in response.")


def nugget_score(judge, nugget, answer):
    """One rubric nugget, BEAM's prompt as one user message: 1.0, 0.5 or 0.0."""
    prompt = UNIFIED_JUDGE_PROMPT.replace("<rubric_item>", nugget).replace("<llm_response>", answer)
    return float(parse_json_response(judge([{"role": "user", "content": prompt}]))["score"])


def equivalent(judge, first, second):
    """BEAM's llm_equivalence. A blank answer line is never the same event, so it costs no call."""
    if not second.strip():
        return False
    messages = [{"role": "system", "content": EQUIVALENCE_SYSTEM},
                {"role": "user", "content": EQUIVALENCE_USER.format(first=first, second=second)}]
    return "yes" in judge(messages).lower()


def event_ordering_score(reference, system, judge):
    """BEAM's event_ordering_score with align_type="llm": each answer line takes the first unused
    reference event it describes; tau-b over the union, unmatched or missing items ranked last."""
    from scipy.stats import kendalltau
    used, system_canon = set(), []
    for line in system:
        match = next((i for i, r in enumerate(reference) if i not in used and equivalent(judge, r, line)), None)
        if match is None:
            system_canon.append(line)
        else:
            system_canon.append(reference[match])
            used.add(match)
    tp = len(set(reference) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference])
    fn = len([x for x in reference if x not in system_canon])
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    union = list(dict.fromkeys(list(reference) + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq):
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    tau_b, _ = kendalltau(to_rank(reference), to_rank(system_canon), variant="b", method="auto")
    tau_norm = (tau_b + 1) / 2 if tau_b is not None and tau_b == tau_b else 0
    return dict(precision=precision, recall=recall, f1=f1, tau_norm=tau_norm, final_score=tau_norm * f1)


def score(ask, answer, judge):
    """judge(messages) -> reply text. BEAM's evaluate_<type>: the mean nugget score; event ordering
    also gets event_ordering_score, and BEAM's report counts its tau_norm (report_results.py)."""
    nuggets = ask["reference"]["nuggets"]
    scores = [nugget_score(judge, n, answer) for n in nuggets]
    out = dict(llm_judge_score=sum(scores) / len(scores) if scores else 0.0, nuggets=scores)
    if ask["type"] == "event_ordering":
        out.update(event_ordering_score(nuggets, answer.split("\n"), judge))
    out["beam_score"] = out["tau_norm"] if ask["type"] == "event_ordering" else out["llm_judge_score"]
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
