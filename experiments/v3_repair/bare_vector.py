"""Independent bare-vector control for Z2: role-tagged original chunks, cosine top-k.

None of ICE's memory mechanisms: no learned gate, reranker, BM25, graph, claims,
summaries, procedures, importance/recency boosts, clusters, decay or cold
exclusion. It uses ICE's embedding model with plain encoding for both question
and chunk, as ICE does. Chunking is deterministic word windows with character
offsets into each original role text. The only post-processing is dropping a
chunk that overlaps one already selected from the same message, and packing to
the matched memory token budget (spec §2.5).

`--tune` picks chunk size and k on the development history only, scoring whether
each question's reviewed evidence spans reach the input (recent turns or
selected chunks): the same span-availability measure used for ICE's arms. It
reads originals and saved baseline contexts; no database, cloud or writer call.
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-v3-bare-vector-1"
GRID = dict(chunk_words=(100, 200, 400, None), k=(10, 20, None))


@dataclass(frozen=True)
class Chunk:
    turn: int
    role: str
    start: int
    end: int
    text: str

    def render(self):
        speaker = "User" if self.role == "user" else "Assistant"
        return f"[{speaker}, turn {self.turn}] {self.text}"


def windows(text, chunk_words, overlap=0.25):
    """Word windows as (start, end) offsets; `None` keeps the whole message."""
    words = [m.span() for m in re.finditer(r"\S+", text)]
    if not words: return []
    if chunk_words is None or len(words) <= chunk_words:
        return [(words[0][0], words[-1][1])]
    step = max(1, chunk_words - int(chunk_words * overlap))
    spans, first = [], 0
    while True:
        last = min(first + chunk_words, len(words)) - 1
        spans.append((words[first][0], words[last][1]))
        if last == len(words) - 1: return spans
        first += step


def chunk_turns(turns, chunk_words):
    """`turns`: dicts with turn_number, prompt, response — originals only."""
    chunks = []
    for turn in turns:
        for role, text in (("user", turn["prompt"]), ("assistant", turn["response"])):
            chunks += [Chunk(turn["turn_number"], role, a, b, text[a:b]) for a, b in windows(text, chunk_words)]
    return chunks


class Index:
    """Append-only embedding cache keyed by chunk text; one encoder for all."""

    def __init__(self, encode):
        self.encode, self.vectors = encode, {}

    def vector(self, texts):
        import numpy as np
        missing = sorted({t for t in texts if t not in self.vectors})
        if missing:
            for text, value in zip(missing, self.encode(missing)):
                value = np.asarray(value, dtype="float32")
                self.vectors[text] = value / (np.linalg.norm(value) or 1.0)
        return np.stack([self.vectors[t] for t in texts])


def select(index, question, chunks, budget_tokens, k, count):
    """Cosine order; skip same-message overlaps; pack until the budget or k."""
    if not chunks: return []
    query = index.vector([question])[0]
    scores = index.vector([c.text for c in chunks]) @ query
    order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].turn, chunks[i].role, chunks[i].start))
    chosen, used = [], 0
    for i in order:
        chunk = chunks[i]
        if any(c.turn == chunk.turn and c.role == chunk.role and c.start < chunk.end and chunk.start < c.end
               for c, _ in chosen):
            continue
        cost = count(chunk.render())
        if used + cost > budget_tokens: continue
        chosen.append((chunk, float(scores[i]))); used += cost
        if k is not None and len(chosen) >= k: break
    return chosen


def recent_turns(stage):
    """Turns already in the prompt's recent block (prior assistant messages)."""
    return sum(1 for m in stage["prompt_messages"][:-1] if m["role"] == "assistant")


def span_available(span, chosen, recent_from):
    if span["turn"] >= recent_from: return True
    covered = sorted((c.start, c.end) for c, _ in chosen if c.turn == span["turn"] and c.role == span["role"])
    reach = span["start"]
    for start, end in covered:
        if start <= reach < end: reach = end
        if reach >= span["end"]: return True
    return False


def tune(originals, baseline, ledger, encode, count, grid=GRID):
    """Span availability per (chunk_words, k) at each question's matched budget."""
    turns = {t["turn_number"]: t for t in originals}
    units = {json.loads(p.read_text())["probe_id"]: json.loads(p.read_text())["probe"]
             for p in sorted((baseline / "units").glob("*.json"))}
    records = {r["probe_id"]: r for r in ledger["records"]}
    if set(records) != set(units): raise ValueError("ledger and baseline cover different questions")
    from functools import lru_cache
    count = lru_cache(maxsize=None)(count)
    index, rows = Index(encode), []
    for chunk_words in grid["chunk_words"]:
        for k in grid["k"]:
            all_spans = probes_complete = spans_covered = tokens = 0
            for key in sorted(units):
                probe = units[key]
                control = probe["controls"]["vector_only"]
                budget = control["context_ledger"]["blocks"]["evidence"]
                split = probe["split_turn"]
                recent_from = split - recent_turns(control) + 1
                older = [turns[n] for n in range(1, recent_from)]
                chosen = select(index, probe["question"], chunk_turns(older, chunk_words), budget, k, count)
                hits = [span_available(s, chosen, recent_from) for s in records[key]["source_spans"]]
                all_spans += len(hits); spans_covered += sum(hits); probes_complete += all(hits)
                tokens += sum(count(c.render()) for c, _ in chosen)
            rows.append(dict(chunk_words=chunk_words, k=k, questions_all_spans=probes_complete,
                             spans_covered=spans_covered, spans=all_spans, memory_tokens=tokens))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--conversation", required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Four-arm baseline contexts (budgets, recent window).")
    parser.add_argument("--ledger", type=Path, required=True, help="Reviewed content-contract ledger (evidence spans).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logs = (ROOT / "logs").resolve()
    if any(not p.resolve().is_relative_to(logs) for p in (args.originals, args.baseline, args.ledger, args.out)):
        raise ValueError("private inputs and outputs stay below logs/")
    from src.memory import tokens
    from src.memory.embedder import get_embedder
    embedder = get_embedder()
    encode = lambda texts: embedder.encode(texts, convert_to_tensor=False, batch_size=8, show_progress_bar=False)
    originals = json.loads(args.originals.read_text())[args.conversation]
    rows = tune(originals, args.baseline.resolve(), json.loads(args.ledger.read_text()), encode, tokens.count)
    inputs = [args.originals, args.baseline / "contexts.json", args.ledger, Path(__file__)]
    result = dict(version="v3", policy=POLICY, development_only=True, rows=rows,
                  input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  measure="reviewed evidence spans present in recent turns or selected chunks; not answer quality")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(f"chunk_words={str(row['chunk_words']):>4} k={str(row['k']):>4}  questions with all spans "
              f"{row['questions_all_spans']:>2}/67  spans {row['spans_covered']}/{row['spans']}  tokens {row['memory_tokens']}")


if __name__ == "__main__":
    main()
