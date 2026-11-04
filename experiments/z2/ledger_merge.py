"""Z2 S2: merge two writers' raw ledgers into one verified ledger (spec D4, D6, §2).

Per window, each writer family checks the OTHER family's entries against the
same turns the writer read. The checker is not asked for a verdict (TRAPS 136:
a "which verdict" question follows something other than the source — MiMo's
supported/unsupported followed whether it had a similar entry of its own,
2026-10-18). It is shown only each entry's subject and sentence, never its
speaker or modality, and reports what the turns show: who says it (user,
assistant, nobody), how (asserted, suggested, hypothetical, not stated), and any
detail of the sentence the turns contradict. Code compares those with the entry:

- observation matches -> verified: two families agree it is in the source;
- otherwise -> a third-family tie-break makes the same observation; a kept entry
  is NOT verified (basis "tie_break") and both outcomes go to the review queue.
  With no tie-breaker (FINAL, Z3 D13: Muse writes no ground truth) a disputed entry
  is unconfirmed ("disputed") — kept on record, never used for questions — and goes
  to the review queue.

Matched entries collapse into one (union of citations, both writers). The
tie-breaker is a parameter so candidates can be compared on the same disputes.
Each model call is one durable, identity-checked line in `merge-raw.jsonl`;
rerunning resumes. Outputs: `ledger.json`, `review-queue.json`.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import ledger as L

POLICY = "ice-z2-ledger-merge-3"
# Identity of a model call: changes only when what is SENT changes, so a merge
# rule change re-merges from saved calls without calling anyone.
CALL_POLICY = "ice-z2-ledger-merge-2"
OUTPUT_TOKENS = 12_000

OBSERVE = """For each entry, read the turns and report what they show:
- said_by: who in the turns says this: "user", "assistant", or "nobody" if neither does
- stated_as: how it is said: "asserted" (stated as true, chosen or done), "suggested"
  (proposed or recommended, not adopted by the user), "hypothetical" (a what-if, an option
  being considered), or "not_stated"
- wrong_detail: a detail in the entry's sentence that the turns contradict or do not contain
  (a name, number, date, place or qualifier), quoted from the sentence; null if every detail
  is in the turns
Judge from the turns only, not from how the sentence is worded."""

VERIFY_SYSTEM = """You check ledger entries that another model extracted from a conversation.
""" + OBSERVE + """
Also give match: the id of the entry under YOURS that records the same item (same
information, any wording), or null if none does.
Return only JSON: {"checks": [{"id": "...", "said_by": "...", "stated_as": "...",
"wrong_detail": "..."|null, "match": "..."|null}]} with one object per CHECK entry."""

TIE_SYSTEM = """Two models disagree about ledger entries extracted from a conversation.
""" + OBSERVE + """
Return only JSON: {"decisions": [{"id": "...", "said_by": "...", "stated_as": "...",
"wrong_detail": "..."|null}]} with one object per entry."""


def observed(entry, row):
    """(supported, reason) from a checker's observation, or (None, reason) when
    the observation is malformed."""
    said, how, wrong = row.get("said_by"), row.get("stated_as"), row.get("wrong_detail")
    if said not in ("user", "assistant", "nobody") or how not in (
            "asserted", "suggested", "hypothetical", "not_stated"):
        return None, "malformed"
    if said == "nobody" or how == "not_stated":
        return False, "not_stated"
    if said != entry["speaker"]:
        return False, "speaker"
    if how != entry["modality"]:
        return False, "modality"
    if isinstance(wrong, str) and wrong.strip():
        return False, "wrong_detail"
    return True, None


def entry_line(e, alias, *, blind=False):
    """One entry as JSON under a short prompt alias; *blind* hides the fields the
    checker must observe."""
    keys = ("subject", "value") if blind else ("kind", "subject", "value", "speaker", "modality")
    return json.dumps({"id": alias, **{k: e[k] for k in keys}}, ensure_ascii=False)


def aliases(entries, prefix):
    """Short ids for the prompt (C01, Y01, D01): two writers' raw ids differ only
    in their first word, and DeepSeek once answered for its own list instead of
    the one it was asked to check (dev windows 1 and 3, 2026-10-18)."""
    return {f"{prefix}{n:02d}": e for n, e in enumerate(entries, 1)}


def turns_text(originals, record):
    return L.render(originals, record["context"] + record["new"])


def verify_prompt(originals, record, check, yours):
    return (f"TURNS:\n{turns_text(originals, record)}\n\n=====\n\n"
            "CHECK (answer for exactly these ids):\n"
            + "\n".join(entry_line(e, a, blind=True) for a, e in check.items())
            + "\n\n=====\n\nYOURS (only for match):\n"
            + ("\n".join(entry_line(e, a) for a, e in yours.items()) or "(none)"))


def tie_prompt(originals, record, disputed):
    return (f"TURNS:\n{turns_text(originals, record)}\n\n=====\n\nENTRIES:\n"
            + "\n".join(entry_line(e, a, blind=True) for a, e in disputed.items()))


def _json(text, key):
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0]
    rows = json.loads(body, strict=False).get(key)
    if not isinstance(rows, list):
        raise ValueError(f"reply has no {key} list")
    return rows


def complete_lines(path):
    """Lines of a JSONL file that parse; a last line cut by a stop mid-write is skipped with a warning."""
    lines = Path(path).read_text().splitlines()
    out = []
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except ValueError:
            if n == len(lines) - 1:
                print(f"WARNING: {path}: last line cut mid-write, skipped (its call is made again)", flush=True)
                continue
            raise
        out.append(line)
    return out


def one_writer(directory, *, poll=30, say=print):
    """Hold <directory>/.writer.lock for this process's life, waiting while another process holds
    it — so a stage started by two runners (the main run and a lane, Z3 2026-11-04) runs once at a
    time, the second resuming from the first's saved calls. Take it BEFORE any Calls is opened on
    the directory: Calls reads the saved calls once, when it is made. Returns the held lock file."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    held = open(directory / ".writer.lock", "a")
    waited = False
    while True:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not waited:
                say(f"another process is writing {directory}; waiting for it to finish, then resuming", flush=True)
                waited = True
            time.sleep(poll)
    if waited:
        say(f"{directory} is free; resuming from its saved calls", flush=True)
    return held


class Calls:
    """Durable, resumable model calls keyed by identity. Safe to share between
    threads: appends are serialised."""

    def __init__(self, path):
        import threading
        self.path = path
        self.done = {}
        self._lock = threading.Lock()
        if path.exists():
            for line in complete_lines(path):
                row = json.loads(line)
                self.done[row["identity"]] = row

    def get(self, identity, run):
        if identity in self.done:
            return self.done[identity]
        started = time.monotonic()
        text, usage = run()
        row = dict(policy=CALL_POLICY, identity=identity, text=text, usage=usage,
                   seconds=round(time.monotonic() - started, 2))
        with self._lock:
            L._append(self.path, row)
            self.done[identity] = row
        return row


def ask(calls, role, meta, generate, system, prompt, key, session, *, attempt=0):
    identity = L.sha256(json.dumps(dict(policy=CALL_POLICY, role=role, meta=meta, system=L.sha256(system),
                                        prompt=L.sha256(prompt), attempt=attempt), sort_keys=True))

    def run():
        result = generate([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                          temperature=0, max_output_tokens=OUTPUT_TOKENS, session_id=session)
        return result.text, result.usage

    row = calls.get(identity, run)
    try:
        return _json(row["text"], key), row
    except (ValueError, json.JSONDecodeError):
        return None, row


def _cut(exc) -> bool:
    """A reply stopped by its output cap (chat: 'length'; Responses API: 'incomplete')."""
    from experiments.lme.cloud_provider import CloudCompletionError
    return isinstance(exc, CloudCompletionError) and any(w in str(exc) for w in ("length", "incomplete"))


def ask_covering(calls, role, meta, generate, system, build, key, session, wanted):
    """Answers keyed by alias; *build(subset)* makes the prompt for a subset of
    *wanted*. A reply missing any wanted alias gets one fresh call. A reply cut by
    the output cap is asked again in halves (Muse's reasoning once used 9,445 of
    12,000 output tokens; the next C call ran out, 2026-10-18); a single entry that
    still does not fit is left out, and the caller counts it unanswered."""
    answers, retried = {}, False
    try:
        for attempt in (0, 1):
            rows, _ = ask(calls, role, meta, generate, system, build(wanted), key, session, attempt=attempt)
            answers = {r.get("id"): r for r in rows or [] if isinstance(r, dict) and r.get("id") in wanted}
            if set(wanted) <= set(answers):
                break
            retried = True
    except Exception as exc:                                  # noqa: BLE001
        if not _cut(exc):
            raise
        if len(wanted) == 1:
            return {}, True
        keys = list(wanted)
        halves = (dict((k, wanted[k]) for k in keys[:len(keys) // 2]),
                  dict((k, wanted[k]) for k in keys[len(keys) // 2:]))
        answers = {}
        for half in halves:
            got, _ = ask_covering(calls, role, meta, generate, system, build, key, session, half)
            answers.update(got)
        retried = True
    return answers, retried


def load_raw(directory, writer):
    return [json.loads(line) for line in (directory / f"ledger-raw-{writer}.jsonl").open()]


def merge(history, raw_dir, out, *, writers, tie_breaker, models):
    """*models* maps role -> (generate, metadata). Returns (ledger, queue)."""
    originals = L.load_history(history)
    a, b = writers
    raw = {w: load_raw(raw_dir, w) for w in writers}
    if [r["new"] for r in raw[a]] != [r["new"] for r in raw[b]]:
        raise ValueError("the two writers' windows differ; merge needs the same windows")
    out.mkdir(parents=True, exist_ok=True)
    calls = Calls(out / "merge-raw.jsonl")
    session = lambda role: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{CALL_POLICY}|{history}|{role}"))
    checks, queue, match_pairs = {}, [], []
    stats = dict(entries=0, matched=0, confirmed_unmatched=0, disputed=0, unanswered=0,
                 retried_calls=0)
    for index, (ra, rb) in enumerate(zip(raw[a], raw[b])):
        disputed = []
        for writer, checker, mine, theirs in ((a, b, ra, rb), (b, a, rb, ra)):
            if not mine["entries"]:
                continue
            generate, meta = models[checker]
            check, yours = aliases(mine["entries"], "C"), aliases(theirs["entries"], "Y")
            answered, retried = ask_covering(calls, checker, meta, generate, VERIFY_SYSTEM,
                                             lambda sub: verify_prompt(originals, mine, sub, yours),
                                             "checks", session(checker), check)
            stats["retried_calls"] += retried
            alias_of = {e["id"]: a for a, e in check.items()}
            for e in mine["entries"]:
                stats["entries"] += 1
                r = answered.get(alias_of[e["id"]])
                supported, reason = observed(e, r) if r is not None else (None, "missing")
                if supported is None:
                    stats["unanswered"] += 1
                    checks[e["id"]] = dict(supported=None, status="unconfirmed",
                                           unconfirmed_reason="check_unanswered", checker=checker)
                    queue.append(dict(kind="unanswered_check", entry=e, checker=checker))
                    continue
                match = yours[r["match"]]["id"] if r.get("match") in yours else None
                checks[e["id"]] = dict(supported=supported, reason=reason, observation=r,
                                       match=match, checker=checker)
                if supported:
                    if match:
                        match_pairs.append((e["id"], match))
                else:
                    disputed.append(e)
        if disputed and tie_breaker is None:
            stats["disputed"] += len(disputed)
            for e in disputed:
                checks[e["id"]].update(status="unconfirmed", unconfirmed_reason="disputed")
                queue.append(dict(kind="disputed", entry=e, checker=checks[e["id"]]["checker"],
                                  checker_reason=checks[e["id"]]["reason"]))
        elif disputed:
            stats["disputed"] += len(disputed)
            generate, meta = models[tie_breaker]
            asked = aliases(disputed, "D")
            by_alias, retried = ask_covering(calls, tie_breaker, meta, generate, TIE_SYSTEM,
                                             lambda sub: tie_prompt(originals, ra, sub), "decisions",
                                             session(tie_breaker), asked)
            stats["retried_calls"] += retried
            decided = {asked[a]["id"]: r for a, r in by_alias.items()}
            for e in disputed:
                check = checks[e["id"]]
                tie = decided.get(e["id"])
                keep, tie_reason = observed(e, tie) if tie is not None else (None, "missing")
                if keep is None:
                    stats["unanswered"] += 1
                    check.update(status="unconfirmed", unconfirmed_reason="tie_break_unanswered")
                    queue.append(dict(kind="unanswered_tie_break", entry=e, tie_breaker=tie_breaker))
                    continue
                check["tie_break"] = dict(model=tie_breaker, observation=tie, reason=tie_reason)
                if keep:
                    check["status"] = "tie_break"
                elif (check["reason"] == tie_reason == "modality"
                      and tie["stated_as"] == check["observation"]["stated_as"]):
                    # Both observers see the item from this speaker with every detail
                    # right, and agree on another modality: the label was wrong, not
                    # the content (maintainer, 2026-10-18).
                    check.update(status="relabeled", modality=tie["stated_as"])
                else:
                    check.update(status="unconfirmed", unconfirmed_reason=tie_reason)
                queue.append(dict(kind="tie_break", entry=e, checker=check["checker"],
                                  checker_reason=check["reason"], tie_breaker=tie_breaker,
                                  tie_observation=tie, status=check["status"]))
    # Nothing is deleted. Entries the checker supported are verified and merge with
    # their matches; every other entry stays on its own with a status saying what
    # it is: tie_break (kept, one family), relabeled (modality corrected), or
    # unconfirmed (+ why). Only verified/tie_break/relabeled entries feed
    # questions; unconfirmed ones mark content that has SOME source, so an answer
    # repeating them is not scored as invented (maintainer, 2026-10-18).
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in match_pairs:
        parent[root(x)] = root(y)
    entries = {e["id"]: e for w in writers for r in raw[w] for e in r["entries"]}
    groups = {}
    for entry_id, e in entries.items():
        c = checks.setdefault(entry_id, dict(supported=None, status="unconfirmed",
                                             unconfirmed_reason="unchecked"))
        if c["supported"]:
            c["status"] = "verified"
        # Separate key spaces: a verified group's root can be the id of an entry
        # that itself stayed unconfirmed.
        key = ("verified", root(entry_id)) if c["status"] == "verified" else ("single", entry_id)
        groups.setdefault(key, []).append(e)
    merged = []
    for members in groups.values():
        members.sort(key=lambda e: (e["writer"], e["id"]))
        head = members[0]
        check = checks[head["id"]]
        status = check["status"]
        writers_in = sorted({m["writer"] for m in members})
        if status == "verified":
            stats["matched" if len(writers_in) == 2 else "confirmed_unmatched"] += 1
        stats[status] = stats.get(status, 0) + 1
        if status == "unconfirmed":
            key = f"unconfirmed_{check['unconfirmed_reason']}"
            stats[key] = stats.get(key, 0) + 1
        citations, seen = [], set()
        for m in members:
            for c in m["citations"]:
                where = (c["turn"], c["role"], c["start"], c["end"])
                if where not in seen:
                    seen.add(where)
                    citations.append(c)
        merged.append(dict(
            id=f"{history}:{head['id']}", history=history, kind=head["kind"], subject=head["subject"],
            value=head["value"], speaker=head["speaker"], modality=check.get("modality", head["modality"]),
            citations=citations, valid_from_turn=min(m["valid_from_turn"] for m in members),
            supersedes_raw=sorted({m["supersedes"] for m in members if m["supersedes"]}),
            writers=writers_in, members=[m["id"] for m in members],
            status=status, verified=status == "verified",
            usable_for_questions=status in ("verified", "tie_break", "relabeled"),
            **({"written_modality": head["modality"]} if status == "relabeled" else {}),
            **({"unconfirmed_reason": check["unconfirmed_reason"]} if status == "unconfirmed" else {})))
    merged.sort(key=lambda e: (e["valid_from_turn"], e["id"]))
    result = dict(policy=POLICY, history=history, writers=list(writers), tie_breaker=tie_breaker,
                  models={role: models[role][1] for role in models}, stats=stats, entries=merged)
    (out / "ledger.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    (out / "review-queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=1) + "\n")
    return result, queue


def main() -> int:
    from experiments.lme.cloud_provider import load_selected_env
    from experiments.z2.profiles import generator, metadata
    load_selected_env()
    parser = argparse.ArgumentParser(description="Z2 S2 ledger merge")
    parser.add_argument("--history", required=True)
    parser.add_argument("--raw", type=Path, required=True, help="directory with both ledger-raw files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tie-breaker", default="muse", help="a role, or none (FINAL: disputes unconfirmed)")
    parser.add_argument("--offline", action="store_true",
                        help="re-merge from saved calls only; any call that is not saved fails")
    args = parser.parse_args()
    from scripts.z1.private_logs import under_logs
    for path in (args.raw, args.out):
        if not under_logs(path):
            raise ValueError("ledger data is private: keep it under logs/")
    tie_breaker = None if args.tie_breaker == "none" else args.tie_breaker
    roles = {"deepseek", "mimo"} | ({tie_breaker} if tie_breaker else set())
    if args.offline:
        def refuse(*_args, **_kwargs):
            raise RuntimeError("--offline: this merge needs a model call that is not saved")
        models = {role: (refuse, metadata(role)) for role in roles}
    else:
        models = {role: (generator(role).generate, metadata(role)) for role in roles}
    result, queue = merge(args.history, args.raw, args.out, writers=("deepseek", "mimo"),
                          tie_breaker=tie_breaker, models=models)
    print(json.dumps(result["stats"]), f"queue {len(queue)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ── S2 change chains ─────────────────────────────────────────────────────

def chain(entries):
    """Validity from the writers' change links (spec D4): an entry whose raw member
    said it supersedes another raw entry closes that entry's validity at its own
    valid_from_turn. Only entries usable for questions may close another (an
    unconfirmed change is not evidence that the old value stopped). An entry is
    never closed by one that starts before it. Returns new dicts; input unchanged."""
    out = [dict(e, valid_until_turn=None, superseded_by=None) for e in entries]
    by_member = {m: e for e in out for m in e["members"]}
    for newer in out:
        if not newer.get("usable_for_questions"):
            continue
        for raw in newer.get("supersedes_raw") or ():
            older = by_member.get(raw)
            if older is None or older is newer or older["valid_from_turn"] > newer["valid_from_turn"]:
                continue
            if older["valid_until_turn"] is None or newer["valid_from_turn"] < older["valid_until_turn"]:
                older["valid_until_turn"] = newer["valid_from_turn"]
                older["superseded_by"] = newer["id"]
    return out


def state_at(entries, checkpoint):
    """Entries current at *checkpoint* (a turn number): started by then, not yet
    replaced by then."""
    return [e for e in entries if e["valid_from_turn"] <= checkpoint
            and (e.get("valid_until_turn") is None or e["valid_until_turn"] > checkpoint)]
