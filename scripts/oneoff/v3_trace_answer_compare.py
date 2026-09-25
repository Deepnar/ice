"""Private paired answer diagnostic for two v3 source-to-prompt traces.

The reports contain personal dialogue. Inputs and output must stay under the
gitignored logs directory. This compares prompt-selected evidence, not a held-out
benchmark, and leaves judging to a separate evidence review.
"""

import argparse
import json
import re
import uuid
from pathlib import Path

from experiments.lme.cloud_provider import PROFILES, TextGenerator, load_selected_env


LOG_ROOT = (Path(__file__).resolve().parents[2] / "logs").resolve()
CONTEXT_PREFIX = "=== RETRIEVED CONTEXT ==="
ACK = "Understood — I have the background context. What would you like to know?"


def private_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(LOG_ROOT):
        raise ValueError("Trace answers and inputs must stay under gitignored logs/")
    return path


def paired_messages(before, after):
    left = [dict(m) for m in before]
    right = [dict(m) for m in after]
    for messages in (left, right):
        messages[0]["content"] = re.sub(
            r"Current date and time \(UTC\): [^\n]+",
            "Current date and time (UTC): 2026-09-25T00:00:00Z.",
            messages[0]["content"], count=1)
    def without_evidence(messages):
        return [m for m in messages
                if not m["content"].startswith(CONTEXT_PREFIX)
                and m["content"] != ACK]
    if without_evidence(left) != without_evidence(right):
        raise ValueError("Paired prompt differs outside retrieved evidence")
    return left, right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--turns", type=int, nargs="+", required=True,
                        help="One-based turn indexes from the trace")
    args = parser.parse_args()
    before = json.loads(private_path(args.before).read_text())
    after = json.loads(private_path(args.after).read_text())
    output = private_path(args.out)
    if before["source"] != after["source"] or before["source_start"] != after["source_start"]:
        raise ValueError("Traces do not share a source window")
    load_selected_env()
    profile = PROFILES["opencode-luna"]
    answerer = TextGenerator(profile)
    result = {"profile": profile.metadata(), "source": before["source"],
              "source_start": before["source_start"], "turns": []}
    for turn_number in args.turns:
        index = turn_number - 1
        original, repaired = before["turns"][index], after["turns"][index]
        if (not original["complete"] or not repaired["complete"]
                or original["question"] != repaired["question"]):
            raise ValueError(f"Turn {turn_number} is incomplete or not paired")
        left, right = paired_messages(original["prompt_messages"],
                                      repaired["prompt_messages"])
        entry = {"turn": turn_number, "question": original["question"],
                 "before_prompt_tokens": original["prompt_tokens"],
                 "after_prompt_tokens": repaired["prompt_tokens"],
                 "answers": {}}
        result["turns"].append(entry)
        for arm, messages in (("before", left), ("after", right)):
            response = answerer.generate(messages, temperature=0,
                                         max_output_tokens=1200,
                                         session_id=f"ice-v3-trace-{uuid.uuid4()}")
            entry["answers"][arm] = {"text": response.text,
                                      "usage": response.usage,
                                      "response_id": response.response_id}
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"turn {turn_number}: paired answers saved", flush=True)


if __name__ == "__main__":
    main()
