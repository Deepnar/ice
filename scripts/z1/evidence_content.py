#!/usr/bin/env python3
"""Offline, source-reviewed content survival; gold never enters runtime selection.

Exact literals are necessary conditions for an exact-value task, not sufficient
semantic support. Other tasks need separate qualified claim review. Legacy
receipts without text remain unobserved. Completed artifacts are immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LITERAL_POLICY = "native_attributed_json_once_v2"
ATTRIBUTED_QUOTE = re.compile(r'The (?:user|assistant|document) said: (?=")')


def rendered_text_units(text):
    """Decode only complete native attributed JSON strings, exactly once.

    Source code containing a literal backslash remains distinct from the
    transport escape for a newline. Neither adjacent roles nor raw surrounding
    text are joined to manufacture a source span.
    """
    decoder = json.JSONDecoder()
    cursor = 0
    while match := ATTRIBUTED_QUOTE.search(text, cursor):
        start = match.end()
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            # Leave malformed/incomplete quotes untouched, not partly decoded.
            yield text[cursor:start]
            cursor = start
            continue
        yield text[cursor:start]
        yield value
        cursor = start + length
    yield text[cursor:]


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def stage_content(stage):
    return {
        "generated": [f for rows in stage["generated_by_leg"].values() for f in rows]
            if "generated_by_leg" in stage else [{"text": None}],
        "ranked": stage.get("ranked_candidates", [{"text": None}]),
        "budgeted": stage.get("budgeted", [{"text": None}]),
        "selected": stage.get("selected", [{"text": None}]),
        "final_input": [{"text": "\n".join(str(m.get("content", ""))
                          for m in stage.get("prompt_messages", []))}]
            if "prompt_messages" in stage else [{"text": None}],
    }


def content_status(fragments, literals):
    texts = [f.get("text") for f in fragments]
    if any(literal in unit for text in texts if isinstance(text, str)
           for unit in rendered_text_units(text) for literal in literals):
        return "retained"
    if any(not isinstance(text, str) for text in texts):
        return "unobserved"
    return "missing"


def audit_probe(probe, review, *, arm="full"):
    if (review.get("reviewed_through_turn") != probe["split_turn"]
            or not review.get("reason", "").strip()):
        raise ValueError("content review needs the frozen cutoff and a source reason")
    stage = probe["preflight"] if arm == "full" else probe["controls"][arm]
    originals = {gold["turn"]: gold for gold in probe["gold_sources"]}
    requirements, observed = [], set()
    for requirement in review["requirements"]:
        identifier = requirement["id"]
        if identifier in observed or requirement.get("kind") != "exact_literal":
            raise ValueError("duplicate or unsupported content requirement")
        observed.add(identifier)
        spans = requirement["alternatives"]
        if not spans:
            raise ValueError("required literal has no reviewed source")
        literals = []
        for span in spans:
            if type(span["turn"]) is not int or span["turn"] > probe["split_turn"]:
                raise ValueError("required literal comes from after the cutoff")
            original = originals[span["turn"]][{"user": "prompt", "assistant": "response"}[span["role"]]]
            start, end = span["start"], span["end"]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(original)
                    or span["source_sha256"] != sha(original)
                    or original[start:end] != span["literal"]):
                raise ValueError("required literal source identity/offset mismatch")
            literals.append(span["literal"])
        states = {}
        for name, fragments in stage_content(stage).items():
            survival = content_status(fragments, literals)
            rendered_hash = sha(json.dumps(fragments, sort_keys=True, ensure_ascii=False))
            # A literal can occur in a negated or unrelated statement. Without
            # context review, retained text has no positive semantic grade.
            semantic = "not_supported" if survival == "missing" else "unreviewed"
            receipt = requirement.get("semantic_review", {}).get(name)
            if receipt:
                if (receipt.get("content_sha256") != rendered_hash
                        or receipt.get("status") not in {"supported", "not_supported", "uncertain"}
                        or not receipt.get("reason", "").strip()
                        or survival == "unobserved"
                        or (survival == "missing" and receipt["status"] == "supported")):
                    raise ValueError("semantic review does not bind the observed content")
                semantic = receipt["status"]
            states[name] = {"literal": survival, "semantic_support": semantic,
                            "content_sha256": rendered_hash}
        requirements.append({"id": identifier, "kind": "exact_literal", "stages": states})
    if not requirements:
        raise ValueError("content review has no required units")
    return {"probe_id": probe["probe_id"], "arm": arm, "literal_policy": LITERAL_POLICY,
            "split_turn": probe["split_turn"], "requirements": requirements,
            "source_id_coverage": stage.get("gold_fragment_coverage"),
            "indexed_and_eligible_content": "unobserved",
            "answer_memory_use": "not_inferred_from_literal_or_answer_correctness"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("--arm", default="full", choices=["full", "no_codex", "vector_only", "recent_only"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logs = (Path(__file__).resolve().parents[2] / "logs").resolve()
    if any(not p.resolve().is_relative_to(logs) for p in (args.trace, args.review, args.out)):
        raise ValueError("content audit artifacts must remain under logs/")
    if args.out.exists():
        raise ValueError("content audit output exists; use a new derived path")
    raw = args.trace.read_bytes()
    packet_bytes = args.review.read_bytes()
    packet = json.loads(packet_bytes)
    if (packet.get("kind") != "v3_required_evidence_review"
            or packet.get("trace_sha256") != hashlib.sha256(raw).hexdigest()):
        raise ValueError("content packet does not bind this trace")
    probes = {r["probe_id"]: r for line in raw.splitlines()
              if (r := json.loads(line)).get("event") == "as_of_probe"}
    seen, records = set(), []
    for review in packet["records"]:
        identifier = review["probe_id"]
        if identifier in seen:
            raise ValueError("duplicate content probe review")
        seen.add(identifier)
        records.append(audit_probe(probes[identifier], review, arm=args.arm))
    result = {"version": "v3", "score_of_record": False, "literal_policy": LITERAL_POLICY,
              "trace_sha256": hashlib.sha256(raw).hexdigest(),
              "review_sha256": hashlib.sha256(packet_bytes).hexdigest(),
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "records": records,
              "limits": ["Exact-value survival only; other semantic tasks require a qualified review.",
                         "Final input includes recent/query text; presence does not establish retrieval's contribution.",
                         "Old stages without text remain unobserved; IDs do not fill that gap."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as sink:
        json.dump(result, sink, sort_keys=True, indent=2)
        sink.write("\n")
    print(f"v3 content audit saved {len(records)} reviewed occurrences to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
