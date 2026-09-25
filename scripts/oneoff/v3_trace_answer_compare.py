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
SUMMARY_SECTION = "\n\n=== CONVERSATION SUMMARY ===\n"
BOOKMARK_SECTION = "\n\n=== BOOKMARKED BY THE USER ===\n"
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
    def without_summary(content):
        start = content.find(SUMMARY_SECTION)
        if start < 0:
            return content
        end = content.find(BOOKMARK_SECTION, start + len(SUMMARY_SECTION))
        return content[:start] + (content[end:] if end >= 0 else "")
    def without_evidence(messages):
        kept = []
        for index, message in enumerate(messages):
            if message["content"].strip().startswith(CONTEXT_PREFIX) or message["content"] == ACK:
                continue
            item = dict(message)
            if index == 0:
                item["content"] = without_summary(item["content"])
            kept.append(item)
        return kept
    if without_evidence(left) != without_evidence(right):
        raise ValueError("Paired prompt differs outside retrieved evidence")
    return left, right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--probe-report", help="One trace with full/no-Codex/vector probe arms")
    parser.add_argument("--out", required=True)
    parser.add_argument("--turns", type=int, nargs="+",
                        help="One-based turn indexes from the trace")
    parser.add_argument("--profile", default="opencode-luna6",
                        choices=("opencode-luna6", "opencode-luna"))
    parser.add_argument("--probe-arms", nargs="+",
                        choices=("full", "no_codex", "vector_only", "no_summary"),
                        help="Probe arms to answer; default is every recorded arm")
    args = parser.parse_args()
    output = private_path(args.out)
    if bool(args.probe_report) == bool(args.before or args.after):
        raise ValueError("Choose either a probe report or a before/after pair")
    load_selected_env()
    profile = PROFILES[args.profile]
    answerer = TextGenerator(profile)
    if args.probe_report:
        source = json.loads(private_path(args.probe_report).read_text())
        result = {"profile": profile.metadata(), "source": source["source"],
                  "source_start": source["source_start"], "probes": []}
        for probe in source.get("probes", []):
            arms = probe["arms"]
            arm_order = ("full", "no_codex", "vector_only") + (
                ("no_summary",) if "no_summary" in arms else ())
            if set(arms) != set(arm_order):
                raise ValueError("Probe report lacks a retrieval arm")
            messages = {"full": paired_messages(
                arms["full"]["prompt_messages"],
                arms["no_codex"]["prompt_messages"])[0]}
            for arm in arm_order[1:]:
                _, messages[arm] = paired_messages(
                    arms["full"]["prompt_messages"], arms[arm]["prompt_messages"])
            entry = {"question": probe["question"], "gold": probe.get("gold"),
                     "source_pair": probe.get("source_pair"), "answers": {}}
            result["probes"].append(entry)
            selected_arms = args.probe_arms or arm_order
            if len(set(selected_arms)) != len(selected_arms) or any(
                    arm not in arm_order for arm in selected_arms):
                raise ValueError("Requested probe arm is duplicate or absent")
            for arm in selected_arms:
                response = answerer.generate(
                    messages[arm], temperature=0, max_output_tokens=768,
                    session_id=f"ice-v3-probe-{uuid.uuid4()}")
                entry["answers"][arm] = {"text": response.text,
                                         "usage": response.usage,
                                         "response_id": response.response_id,
                                         "prompt_tokens": arms[arm]["prompt_tokens"]}
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"probe source pair {entry['source_pair']}: {len(selected_arms)} answers saved",
                  flush=True)
        return

    if not (args.before and args.after and args.turns):
        raise ValueError("Before/after mode needs both reports and turn indexes")
    before = json.loads(private_path(args.before).read_text())
    after = json.loads(private_path(args.after).read_text())
    if before["source"] != after["source"] or before["source_start"] != after["source_start"]:
        raise ValueError("Traces do not share a source window")
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
