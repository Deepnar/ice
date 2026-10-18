"""Z2 fact ledger (spec Z2_EVALUATION D4, §2): citations first.

A citation is ``{turn, role, start, end, quote, source_sha256}``: *quote* is
``role_text[start:end]`` of that turn's original ``user`` (prompt) or
``assistant`` (response) text, and *source_sha256* is the sha256 of that whole
role text. Writers supply turn, role and an exact quote; `resolve` finds the
offsets and the hash, so no model is trusted with character arithmetic.

Locating tolerates whitespace only: a writer that flattens a list's line breaks
or a table's tabs into spaces still finds its span (13 of 17 failed quotes in
the first dev window were that, 2026-10-17). The STORED quote is the original's
own bytes at the found offsets, never the writer's text, so the check below is
still byte-for-byte. Words, spelling, case and order must match: a corrected
typo or a quote stitched from non-adjacent lines is not found.

`check` is the mechanical citation check every ledger entry and occurrence
passes: the span exists byte-for-byte at its offsets, the turn is at or before
the checkpoint, and the hash matches the original. A failure removes the item
and is counted; it is never repaired by editing the model's output.
"""

import hashlib
import re

ROLES = {"user": "prompt", "assistant": "response"}


def role_text(originals, turn, role):
    """The original text of *role* in pair *turn* (1-based)."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    if type(turn) is not int or not 1 <= turn <= len(originals):
        raise ValueError(f"turn {turn!r} outside the history")
    row = originals[turn - 1]
    if row["turn_number"] != turn:
        raise ValueError("originals are not in turn order")
    return row[ROLES[role]]


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve(originals, turn, role, quote):
    """A full citation for an exact *quote*, or None when it is not in that text.
    The first occurrence is taken; `occurrences` says whether that was a choice."""
    if not isinstance(quote, str) or not quote.strip():
        return None
    try:
        text = role_text(originals, turn, role)
    except ValueError:
        return None
    pattern = re.compile(r"\s+".join(re.escape(word) for word in quote.split()))
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    start, end = matches[0].span()
    return dict(turn=turn, role=role, start=start, end=end, quote=text[start:end],
                source_sha256=sha256(text), occurrences=len(matches))


def problem(originals, citation, checkpoint):
    """Why *citation* fails the check at *checkpoint*, or None when it passes."""
    try:
        turn, role = citation["turn"], citation["role"]
        start, end, quote = citation["start"], citation["end"], citation["quote"]
        digest = citation["source_sha256"]
    except (KeyError, TypeError):
        return "malformed"
    if type(checkpoint) is not int or type(start) is not int or type(end) is not int:
        return "malformed"
    if type(turn) is not int or turn < 1:
        return "malformed"
    if turn > checkpoint:
        return "future_turn"
    try:
        text = role_text(originals, turn, role)
    except ValueError:
        return "no_such_source"
    if digest != sha256(text):
        return "source_hash"
    if not quote or not 0 <= start < end <= len(text):
        return "offsets"
    if text[start:end] != quote:
        return "span_mismatch"
    return None


def check(originals, items, checkpoint_of, citations_of=lambda item: item["citations"]):
    """Split *items* into (kept, removed). An item with no citations, or any
    failing citation, is removed whole; *removed* rows say which and why."""
    kept, removed = [], []
    for item in items:
        cited = citations_of(item) or []
        reasons = [problem(originals, c, checkpoint_of(item)) for c in cited] or ["uncited"]
        failed = [(i, r) for i, r in enumerate(reasons) if r]
        if failed:
            removed.append(dict(item=item, failures=failed))
        else:
            kept.append(item)
    return kept, removed


# ── S1: window extraction ────────────────────────────────────────────────

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -1 (2026-10-18): the B/C ledgers in logs/z2-ledger-2026-10-18-r1 ran under it; its
# SYSTEM prompt (sha256 771a5616...) named a history-B character as the subject example.
# -2: the same prompt with a neutral example. Windows bind the SYSTEM hash, so a -1 run
# never resumes under -2.
POLICY = "ice-z2-ledger-extract-2"
KINDS = ("fact", "preference", "decision", "plan", "advice", "procedure", "event")
MODALITIES = ("asserted", "suggested", "hypothetical")
CHANGES = ("correction", "reversal", "update")
WINDOW_PAIRS, OVERLAP_PAIRS = 25, 5
NEW_TEXT_BUDGET = 48_000        # tokens of new turns per call; larger groups split
PRIOR_BUDGET = 16_000           # tokens of the current ledger shown per call
CONTEXT_BUDGET = 12_000         # tokens of context pairs; C has single pairs of ~27K tokens
OUTPUT_TOKENS = 16_000
INPUTS = ROOT / "logs/v3-offline-input-plan-2026-10-10-r4"

SYSTEM = """You build a fact ledger from a conversation between a user and an AI assistant.
The ledger is ground truth for memory questions, so it must be complete, exact and cited.

Read the NEW turns. Record every durable item stated in them: something that could matter
later in the conversation. That includes facts about the user, other people, projects and
things; preferences, likes, dislikes and constraints; decisions; plans; advice and procedures
the assistant gave; events that happened; and the content of creative or technical work being
built (characters and their traits, rules of a world, design choices). Skip greetings, filler,
one-off reactions, and general knowledge that is not tied to this conversation.

CONTEXT turns are shown only so you understand the new ones. Never record an item that is
stated only in CONTEXT turns; they were processed already.

Each entry:
- kind: one of fact, preference, decision, plan, advice, procedure, event
- subject: a short name for what the entry is about ("a character's power", "user's job",
  "database choice")
- value: one self-contained sentence that states the item fully, understandable without the
  conversation; keep qualifiers (dates, conditions, "maybe", "for now")
- speaker: user or assistant: who said it
- modality: asserted (stated as true or chosen), suggested (proposed, not adopted), or
  hypothetical (a what-if, an option being brainstormed). An assistant proposal the user has
  not accepted is suggested; when the user accepts it, record a separate user entry
- citations: 1 to 3 objects {turn, role, quote}: role is user or assistant; quote is an exact,
  contiguous excerpt of 3 to 40 words copied character for character from that turn,
  including its spelling mistakes and punctuation. Never fix a typo, never skip words or
  lines, never join pieces from different places: use separate citations instead. Cite NEW
  turns
- supersedes: if this entry changes an entry in CURRENT LEDGER (a correction, reversal or
  update of the same subject), that entry's id; otherwise null
- change: correction, reversal or update when supersedes is set; otherwise null

Return only JSON: {"entries": [ ... ]}. Return {"entries": []} if nothing durable is new."""


def load_history(history):
    """Originals for B, C or a development history (dev-<slug>), in turn order."""
    if history in ("B", "C"):
        return json.loads((INPUTS / f"final-originals-{history}.json").read_text())
    if history.startswith("dev-"):
        groups = json.loads((INPUTS / "development-originals.json").read_text())
        return groups[history.removeprefix("dev-")]
    raise ValueError(f"unknown history {history!r}")


def render(originals, turns):
    return "\n\n".join(f"[turn {t} | {role}]\n{role_text(originals, t, role)}"
                       for t in turns for role in ("user", "assistant"))


def pair_tokens(originals, turn):
    from src.memory import tokens
    return tokens.count(render(originals, [turn]))


def windows(originals, *, size=WINDOW_PAIRS, overlap=OVERLAP_PAIRS, budget=NEW_TEXT_BUDGET):
    """Groups of new turns (first `size`, then `size - overlap`), each with the
    `overlap` pairs before it as context. A group over *budget* new-text tokens
    is split into consecutive smaller groups (one pair at minimum)."""
    n, groups, start = len(originals), [], 1
    while start <= n:
        span = size if start == 1 else size - overlap
        groups.append(list(range(start, min(n, start + span - 1) + 1)))
        start += span
    out = []
    for group in groups:
        current, used = [], 0
        for turn in group:
            cost = pair_tokens(originals, turn)
            if current and used + cost > budget:
                out.append(current)
                current, used = [], 0
            current.append(turn)
            used += cost
        out.append(current)
    return [dict(index=i, new=g, context=fit_context(originals, list(range(max(1, g[0] - overlap), g[0]))))
            for i, g in enumerate(out)]


def fit_context(originals, turns, budget=CONTEXT_BUDGET):
    """The latest of *turns* whose pairs fit *budget* tokens (possibly none)."""
    kept, used = [], 0
    for turn in reversed(turns):
        cost = pair_tokens(originals, turn)
        if used + cost > budget:
            break
        kept.insert(0, turn)
        used += cost
    return kept


def _words(text):
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def prior_block(current, window_text, budget=PRIOR_BUDGET):
    """Current entries as one line each. Over budget: the 100 most recent plus
    those sharing the most words with the window, in ledger order."""
    from src.memory import tokens
    line = lambda e: f"{e['id']} | {e['subject']} | {e['value']}"
    lines = [line(e) for e in current]
    if tokens.count("\n".join(lines)) <= budget:
        return "\n".join(lines), len(current), len(current)
    window = _words(window_text)
    recent = set(range(max(0, len(current) - 100), len(current)))
    ranked = sorted((i for i in range(len(current)) if i not in recent),
                    key=lambda i: -len(window & _words(current[i]["subject"] + " " + current[i]["value"])))
    chosen, used = set(recent), tokens.count("\n".join(lines[i] for i in recent))
    for i in ranked:
        cost = tokens.count(lines[i]) + 1
        if used + cost > budget:
            break
        chosen.add(i)
        used += cost
    return "\n".join(lines[i] for i in sorted(chosen)), len(chosen), len(current)


def user_prompt(originals, window, prior_text):
    parts = [f"CURRENT LEDGER (id | subject | value):\n{prior_text or '(empty)'}"]
    if window["context"]:
        parts.append("CONTEXT turns (already processed; do not record from these):\n"
                     + render(originals, window["context"]))
    parts.append(f"NEW turns {window['new'][0]}-{window['new'][-1]}:\n"
                 + render(originals, window["new"]))
    return "\n\n=====\n\n".join(parts)


def parse(text):
    """The entries list from a model reply; raises ValueError when it is not JSON."""
    body = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, re.S)
    if fence:
        body = fence.group(1)
    data = json.loads(body)
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError("reply has no entries list")
    return entries


def admit(originals, raw, window, known_ids, entry_id):
    """A validated, citation-resolved entry, or (None, reason). Never repaired."""
    if not isinstance(raw, dict):
        return None, "not_an_object"
    kind, modality = raw.get("kind"), raw.get("modality")
    speaker, subject, value = raw.get("speaker"), raw.get("subject"), raw.get("value")
    if kind not in KINDS or modality not in MODALITIES or speaker not in ROLES:
        return None, "bad_field"
    if not (isinstance(subject, str) and subject.strip() and isinstance(value, str) and value.strip()):
        return None, "bad_field"
    supersedes, change = raw.get("supersedes"), raw.get("change")
    if supersedes is not None and supersedes not in known_ids:
        return None, "supersedes_unknown"
    if (supersedes is None) != (change is None) or (change is not None and change not in CHANGES):
        return None, "bad_change"
    cited = raw.get("citations")
    if not isinstance(cited, list) or not 1 <= len(cited) <= 3:
        return None, "uncited"
    allowed = set(window["new"]) | set(window["context"])
    resolved = []
    for c in cited:
        if not isinstance(c, dict) or c.get("turn") not in allowed:
            return None, "citation_outside_window"
        r = resolve(originals, c.get("turn"), c.get("role"), c.get("quote"))
        if r is None:
            return None, "quote_not_found"
        resolved.append(r)
    if not any(r["turn"] in window["new"] for r in resolved):
        return None, "context_only"
    return dict(id=entry_id, kind=kind, subject=subject.strip(), value=value.strip(),
                speaker=speaker, modality=modality, citations=resolved,
                valid_from_turn=max(r["turn"] for r in resolved if r["turn"] in window["new"]),
                supersedes=supersedes, change=change), None


def current_ledger(entries):
    """Entries not superseded, in order."""
    replaced = {e["supersedes"] for e in entries if e["supersedes"]}
    return [e for e in entries if e["id"] not in replaced]


def window_identity(history, writer_meta, originals, window, prior_text):
    return sha256(json.dumps(dict(
        policy=POLICY, system=sha256(SYSTEM), history=history, writer=writer_meta,
        output_tokens=OUTPUT_TOKENS, new=window["new"], context=window["context"],
        pairs=[originals[t - 1]["original_pair_sha256"] for t in window["context"] + window["new"]],
        prior=sha256(prior_text)), sort_keys=True))


def _append(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def call_writer(generate, originals, window, prior_text, session):
    """One window, split in halves while the reply is cut by the output cap or
    the provider refuses the input as longer than its context."""
    from experiments.lme.cloud_provider import CloudCompletionError
    try:
        started = time.monotonic()
        result = generate([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user_prompt(originals, window, prior_text)}],
                          temperature=0, max_output_tokens=OUTPUT_TOKENS, session_id=session)
        return [dict(new=window["new"], text=result.text, usage=result.usage,
                     seconds=round(time.monotonic() - started, 2))]
    except Exception as exc:                          # noqa: BLE001
        cut = isinstance(exc, CloudCompletionError) and "length" in str(exc)
        too_long = (getattr(exc, "status_code", None) == 400
                    and any(w in str(exc).lower() for w in ("context", "too long", "maximum", "token")))
        if not (cut or too_long) or len(window["new"]) == 1:
            raise
        half = len(window["new"]) // 2
        first = dict(window, new=window["new"][:half])
        second = dict(window, new=window["new"][half:],
                      context=fit_context(originals, (window["context"] + window["new"][:half])[-OVERLAP_PAIRS:]))
        return (call_writer(generate, originals, first, prior_text, session)
                + call_writer(generate, originals, second, prior_text, session))


def run_writer(history, writer, out, *, limit=None, generate=None, writer_meta=None):
    """Extract *history* with one writer into `out/ledger-raw-{writer}.jsonl`.
    Resumable: an existing line whose identity matches is kept and skipped; one
    that differs stops the run (use a new directory)."""
    originals = load_history(history)
    if generate is None:
        from experiments.z2.profiles import generator, metadata
        generate = generator(writer).generate          # loads the provider env first
        writer_meta = metadata(writer)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ledger-raw-{writer}.jsonl"
    done = [json.loads(line) for line in path.open()] if path.exists() else []
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{POLICY}|{history}|{writer}"))
    plan = windows(originals)[:limit] if limit else windows(originals)
    entries = []        # the ledger as it stood before each window, rebuilt in order
    for window in plan:
        prior_text, shown, total = prior_block(current_ledger(entries), render(originals, window["new"]))
        identity = window_identity(history, writer_meta, originals, window, prior_text)
        if window["index"] < len(done):
            if done[window["index"]]["identity"] != identity:
                raise RuntimeError(f"window {window['index']} differs from the saved run; use a new directory")
            entries.extend(done[window["index"]]["entries"])
            continue
        parts = call_writer(generate, originals, window, prior_text, session)
        known = {e["id"] for e in entries}
        kept, dropped, failures = [], [], []
        for part_index, part in enumerate(parts):
            part_window = dict(window, new=part["new"])
            try:
                raws = parse(part["text"])
            except (ValueError, json.JSONDecodeError) as exc:
                # One fresh call for an unparseable reply; never edit the reply.
                failures.append(dict(part=part_index, error=type(exc).__name__, text=part["text"]))
                retry = call_writer(generate, originals, part_window, prior_text, session)
                parts[part_index] = part = dict(retry[0], retried=True) if len(retry) == 1 else part
                try:
                    raws = parse(part["text"]) if len(retry) == 1 else []
                except (ValueError, json.JSONDecodeError) as again:
                    failures.append(dict(part=part_index, error=type(again).__name__, retry=True))
                    continue
            for n, raw in enumerate(raws):
                entry_id = f"{writer}:{history}:w{window['index']}.{part_index}.{n}"
                entry, reason = admit(originals, raw, part_window, known, entry_id)
                if entry is None:
                    dropped.append(dict(reason=reason, raw=raw))
                else:
                    kept.append(dict(entry, history=history, writer=writer))
        record = dict(policy=POLICY, history=history, writer=writer, writer_meta=writer_meta,
                      index=window["index"], new=window["new"], context=window["context"],
                      identity=identity, prior_shown=shown, prior_total=total,
                      parts=[dict(new=p["new"], text=p["text"], usage=p["usage"], seconds=p["seconds"],
                                  retried=p.get("retried", False)) for p in parts],
                      parse_failures=failures, entries=kept, dropped=dropped)
        _append(path, record)
        done.append(record)
        entries.extend(kept)
        print(f"{writer} {history} window {window['index'] + 1}/{len(plan)} turns "
              f"{window['new'][0]}-{window['new'][-1]}: kept {len(kept)}, dropped {len(dropped)}, "
              f"parse failures {len(failures)}", flush=True)
    return path


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Z2 S1 ledger extraction (one writer)")
    parser.add_argument("--history", required=True, help="B, C or dev-<slug>")
    parser.add_argument("--writer", required=True, choices=("deepseek", "mimo"))
    parser.add_argument("--out", type=Path, required=True, help="private run directory under logs/")
    parser.add_argument("--limit", type=int, help="first N windows only (qualification)")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("ledger output is private: put it under logs/")
    run_writer(args.history, args.writer, args.out, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
