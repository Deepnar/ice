"""Shared plumbing for the B1 pipeline stages.

Every stage reads JSONL, writes JSONL, and must survive being interrupted — a
25k-row labeling pass over two models is hours of GPU time, and losing it to a
crash at row 24,000 is not acceptable. So resume-by-id is built in here rather
than re-implemented (slightly differently, with slightly different bugs) eight
times.

Row shape flowing through the pipeline:

    {"id": str,                 # stable, content-derived — resume + dedupe key
     "source": str,             # lmsys | wildchat | sharegpt | personal | icedev | synth
     "provider": str | None,    # chatgpt | claude | deepseek | gemini (personal rows)
     "text": str,               # the user turn being classified
     "context_text": str|None,  # prior-turn block, ALREADY truncated to budget
     "conversation_id": str|None,
     "turn_index": int|None,
     "ts": str|None,            # ISO timestamp when known (temporal weak supervision)
     "meta": {...}}             # per-stage extras

Labels are added by label.py under a separate ``labels`` key so the corpus text
and its labels never get confused — B1 reuses v1's TEXT but not its LABELS
(trap 1), and keeping them in different keys makes that mechanical.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Iterable, Iterator, Optional

# Repo root on the path — these are standalone scripts, matching house style.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ...and run FROM the repo root. Settings carry repo-relative paths
# (`classifier_model_path`, `label_schema_path`, `models/`), but these stages are
# normally invoked from this directory, so without this every settings-derived
# path resolves against the wrong place — and does so hours into a run, when a
# late stage finally reads one. Stage defaults are absolute, so this only affects
# relative paths a caller types by hand.
os.chdir(ROOT)

DATA_DIR = os.path.join(ROOT, "data", "labeled", "v2")
SIMULATION_DIR = os.path.join(ROOT, "data", "simulation")
LEGACY_CORPUS = os.path.join(ROOT, "data", "labeled", "labeled_prompts.jsonl")

# Stage artifacts (named so the flow is readable from an `ls`).
CORPUS_RAW = os.path.join(DATA_DIR, "corpus_raw.jsonl")
CORPUS_SYNTH = os.path.join(DATA_DIR, "corpus_synth.jsonl")
ICEDEV_STITCHED = os.path.join(DATA_DIR, "icedev_stitched.jsonl")
LABELS_A = os.path.join(DATA_DIR, "labels_a.jsonl")
LABELS_B = os.path.join(DATA_DIR, "labels_b.jsonl")
LABELS_TIEBREAK = os.path.join(DATA_DIR, "labels_tiebreak.jsonl")
LABELS_FINAL = os.path.join(DATA_DIR, "labels_final.jsonl")
REVIEW_QUEUE = os.path.join(DATA_DIR, "review_queue.jsonl")
AUDIT_SAMPLE = os.path.join(DATA_DIR, "audit_sample.jsonl")
TRAIN_SPLIT = os.path.join(DATA_DIR, "train.jsonl")
VAL_SPLIT = os.path.join(DATA_DIR, "val.jsonl")
TEST_SPLIT = os.path.join(DATA_DIR, "test.jsonl")

ONLINE_SOURCES = ("lmsys", "wildchat", "sharegpt")
PERSONAL_SOURCES = ("personal", "icedev")

# Sources whose rows are single-turn logs from strangers — the labeling rubric
# raises its memory-evidence threshold for these (legacy STEP 0 / trap 6).
ZERO_CONTEXT_SOURCES = set(ONLINE_SOURCES)


def ensure_data_dir() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def stable_id(source: str, text: str, extra: str = "") -> str:
    """Content-derived id: re-running a stage produces the same ids, so resume
    and cross-stage joins work without a counter that shifts when a filter
    changes."""
    digest = hashlib.sha256(f"{source}\x00{text}\x00{extra}".encode()).hexdigest()
    return f"{source}_{digest[:16]}"


def read_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def completed_ids(path: str) -> set:
    """Ids already present in an output file — the resume key."""
    return {row["id"] for row in read_jsonl(path) if "id" in row}


class JsonlAppender:
    """Append rows, flushing every write.

    Flushing per row is the point: an interrupted labeling run must leave every
    row it finished on disk, not in a buffer.
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self.count = 0

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ── coarse topical bucketing, for SAMPLING ONLY ─────────────────────────────
# Not labels. extract.py uses these to keep the online pull from being 60%
# "help me with my code" — thin buckets get over-sampled. The real topic labels
# come from the labelers.
_BUCKET_CUES = {
    "software": ("code", "python", "javascript", "bug", "error", "function", "api",
                 "server", "docker", "git", "database", "sql", "install", "compile"),
    "stem": ("equation", "physics", "chemistry", "calculus", "theorem", "molecule",
             "biology", "integral", "proof", "algebra"),
    "business": ("marketing", "startup", "invoice", "salary", "customer", "revenue",
                 "business", "client", "resume", "interview", "stock", "tax"),
    "creative": ("story", "character", "novel", "poem", "song", "roleplay", "fantasy",
                 "chapter", "plot", "lyrics", "art", "game design"),
    "admin": ("schedule", "calendar", "email", "meeting", "checklist", "plan my",
              "organise", "organize", "reminder", "todo"),
    "health": ("workout", "diet", "calories", "symptom", "sleep", "doctor", "recipe",
               "travel", "exercise", "medication"),
    "social": ("girlfriend", "boyfriend", "friend", "relationship", "family", "coworker",
               "argument", "date", "apologize"),
    "world": ("election", "war", "history", "country", "president", "economy", "news",
              "culture", "government", "climate"),
    "meta": ("you said", "your memory", "as an ai", "chatgpt", "claude", "your training",
             "previous conversation", "remember when"),
}


def coarse_bucket(text: str) -> str:
    """Cheap keyword bucket for diversity sampling. NEVER a label."""
    lowered = text.lower()
    best, best_hits = "general", 0
    for bucket, cues in _BUCKET_CUES.items():
        hits = sum(1 for cue in cues if cue in lowered)
        if hits > best_hits:
            best, best_hits = bucket, hits
    return best


def is_usable_prompt(text: Optional[str], min_chars: int = 12,
                     max_chars: int = 8000) -> bool:
    """Filter obvious junk before it costs a labeling call.

    The cap matters: a 40k-character pasted document is one row that would blow
    the labeler's context window and teach the classifier nothing its 500-word
    truncation will ever see at inference.
    """
    if not text:
        return False
    stripped = text.strip()
    return min_chars <= len(stripped) <= max_chars
