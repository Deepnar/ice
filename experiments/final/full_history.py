"""FINAL's full-history upper bound (pre-registration r2 §4, FINAL_experiments.md §2.1 item 17).

On BEAM's 500K tier (5 conversations, ~320K–857K tokens — beyond the 256K long-context window)
the answerer, Muse 1.2, answers every question from the ENTIRE conversation, within its own
1,048,576-token window (maintainer, 2026-10-28). Every other condition is answered by the same
model, so the comparison with ICE and long context on the same questions comes straight from
BEAM's main grades — nothing is answered twice.

Built exactly like the long_context condition (experiments/z2/read_passes.py): recent_only's
envelope for the same question, the older history rendered as the same chunks in the evidence
position — every chunk, where long_context keeps only the newest that fit its budget. All of a
conversation's prompts share one cache session, so the provider can reuse the cached
transcript across its questions. A prompt that would not fit WINDOW with the output cap is
recorded in too-long.json, never truncated.

    uv run --no-sync python experiments/final/full_history.py --history BEAM-500K-<id> \
        --read logs/final-run-main/local-read-BEAM-500K-<id> --out logs/final-run-main/upper-read-BEAM-500K-<id> \
        --window 1048576
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-final-full-history-1"
OUTPUT_RESERVE = 16_000            # answers.py's output cap for the cloud answerer


def full_messages(recent_stage, originals, question, checkpoint):
    """recent_only's envelope with every older turn as long_context's chunks, oldest first."""
    from experiments.v3_repair.bare_vector import chunk_turns
    from experiments.z2.read_passes import bare_messages
    recent_from = checkpoint - sum(1 for m in recent_stage["prompt_messages"][:-1] if m["role"] == "assistant") + 1
    chunks = chunk_turns([r for r in originals if r["turn_number"] < recent_from], None)
    chosen = [(c, None) for c in sorted(chunks, key=lambda c: (c.turn, c.role != "user"))]
    return bare_messages(recent_stage, chosen, question), chosen


def build(history, read_dir, out_dir, window, count=None):
    """Writes units/full_history/ (the whole conversation) under *out_dir*; returns a summary."""
    from experiments.z2.ledger import load_history
    from experiments.z2.read_passes import unit_path, write_unit
    if count is None:
        from src.memory.tokens import count_messages as count
    originals = load_history(history)
    read_dir, out_dir = Path(read_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = f"ice-final-upper-{history}"
    built, too_long, sizes = 0, [], []
    for path in sorted((read_dir / "units" / "recent_only").glob("*.json")):
        recent = json.loads(path.read_text())
        ask = recent["ask"]
        if ask.get("kind") != "question":
            continue
        messages, chosen = full_messages(recent["stage"], originals, ask["question"], recent["checkpoint"])
        tokens = count(messages)
        sizes.append(tokens)
        if tokens + OUTPUT_RESERVE > window:
            too_long.append(dict(ask=ask["id"], prompt_tokens=tokens, window=window))
            continue
        write_unit(unit_path(out_dir, "full_history", ask["id"]),
                   dict(policy=POLICY, history=history, condition="full_history", ask=ask,
                        checkpoint=recent["checkpoint"], question_time=recent.get("question_time"),
                        stage=dict(prompt_messages=messages, prompt_tokens=tokens, cache_session=session,
                                   selected=[dict(turn=c.turn, role=c.role) for c, _ in chosen])))
        built += 1
    summary = dict(policy=POLICY, history=history, window=window, built=built, too_long=len(too_long),
                   max_prompt_tokens=max(sizes) if sizes else None)
    (out_dir / "too-long.json").write_text(json.dumps(too_long, indent=1) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", required=True)
    parser.add_argument("--read", type=Path, required=True, help="ICE's read folder for this history")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, required=True, help="the chosen model's context window (tokens)")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("prompts carry the conversation: keep them under logs/")
    print(json.dumps(build(args.history, args.read, args.out, args.window)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
