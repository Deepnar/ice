"""FINAL private histories as Z2-format originals (FINAL_experiments.md §2.1).

From the v1 corpus (data/simulation/simulation_full.jsonl, gitignored) through
experiments/v3_repair/prepare_inputs.py's loader, so rows carry the same fields and hashes as
Z2's B and C. Output (private, gitignored): logs/final-private-inputs/final-originals-<KEY>.json
and a manifest with the corpus fingerprint.

ICE-2 (added 2026-10-31, FINAL_PREREGISTRATION revision 4) is not in that corpus: the v1 corpus was
cut from an earlier DeepSeek export that held only its first 17 pairs. It comes from the later export
(data/simulation/deepseek.json, gitignored) through the paper corpus's own extractor
(scripts/oneoff/extract_deepseek.py) — which rebuilds C's 325 pairs from the same export exactly, so
both histories are paired the same way — and is checked the same way load_corpus checks a history.

    uv run --no-sync python experiments/final/private_originals.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "data/simulation/simulation_full.jsonl"
OUT = ROOT / "logs/final-private-inputs"
# key -> (conversation id prefix, pairs). LAP: the laptop-decision chat (the maintainer approved
# it 2026-10-26 in place of a DeepSeek thread that turned out to be an excluded conversation).
# D (ecc64aab, 251 pairs) was dropped from FINAL 2026-10-28 (the v3 hyper-parameter tuning history).
HISTORIES = {"A": ("633e26f8", 290), "SW": ("bd94e64b", 128), "LAP": ("942699a4", 101)}
EXPORT = ROOT / "data/simulation/deepseek.json"
# key -> (export chat title, conversation id prefix, pairs); the parity chat rebuilds C from the export.
EXPORT_HISTORIES = {"ICE2": ("ICE-2", "cb1ff23b", 331)}
PARITY = ("ICE-1", "a77c15cf", 325)


def export_pairs(conversations, title, prefix, count):
    """One export chat as Z2-format originals, paired exactly as the v1 corpus was."""
    from datetime import datetime

    from experiments.v3_repair.prepare_inputs import digest
    from scripts.oneoff.extract_deepseek import extract_conversation
    chats = [c for c in conversations if c.get("title", "").strip() == title]
    if len(chats) != 1 or not chats[0]["id"].startswith(prefix):
        raise ValueError(f"export chat {title} not found once with id {prefix}")
    msgs, rows, i = extract_conversation(chats[0]), [], 0
    while i < len(msgs) - 1:                                    # the extractor's own pairing loop
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            if msgs[i]["text"] and msgs[i + 1]["text"]:
                row = dict(prompt=msgs[i]["text"], response=msgs[i + 1]["text"],
                           timestamp=msgs[i]["inserted_at"], conversation_id=chats[0]["id"])
                rows.append(dict(row, turn_number=len(rows) + 1, corpus_line=None,
                                 original_pair_sha256=digest(row), timestamp_provenance="synthetic_raw_import"))
            i += 2
        else:
            i += 1
    stamps = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    if len(rows) != count:
        raise ValueError(f"{title}: {len(rows)} pairs, expected {count}")
    if any(s.tzinfo is None for s in stamps) or any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f"{title}: unknown or nonordered source clock")
    return rows


def main() -> int:
    from experiments.v3_repair.prepare_inputs import fingerprint, load_corpus
    groups = load_corpus(CORPUS, {prefix: n for prefix, n in HISTORIES.values()})
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(corpus=fingerprint(CORPUS), histories={})
    for key, (prefix, n) in HISTORIES.items():
        rows = groups[prefix]
        path = OUT / f"final-originals-{key}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
        manifest["histories"][key] = dict(prefix=prefix, pairs=len(rows), file=path.name,
                                          first=rows[0]["timestamp"], last=rows[-1]["timestamp"])
        print(f"{key}: {len(rows)} pairs, {rows[0]['timestamp'][:10]}..{rows[-1]['timestamp'][:10]}")
    conversations = json.loads(EXPORT.read_text())
    parity = export_pairs(conversations, *PARITY)
    corpus_c = load_corpus(CORPUS, {PARITY[1]: PARITY[2]})[PARITY[1]]
    if [r["original_pair_sha256"] for r in parity] != [r["original_pair_sha256"] for r in corpus_c]:
        raise ValueError("the export no longer rebuilds C's corpus pairs; ICE-2 would be paired differently")
    manifest["export"] = dict(fingerprint(EXPORT), parity=f"{PARITY[0]} rebuilds C's {PARITY[2]} corpus pairs")
    for key, (title, prefix, n) in EXPORT_HISTORIES.items():
        rows = export_pairs(conversations, title, prefix, n)
        path = OUT / f"final-originals-{key}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
        manifest["histories"][key] = dict(prefix=prefix, pairs=len(rows), file=path.name, source=title,
                                          first=rows[0]["timestamp"], last=rows[-1]["timestamp"])
        print(f"{key}: {len(rows)} pairs, {rows[0]['timestamp'][:10]}..{rows[-1]['timestamp'][:10]} (export)")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
