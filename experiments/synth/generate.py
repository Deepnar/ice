"""FINAL synthetic set: turn-by-turn generation from a validated plan.

For each turn a simulated USER writes the message from the persona, the session's style
card and beats, the recent exchange, and the planted items due that turn (introduce /
change / re-mention a fact, paste material containing it, tell an event, reject an earlier
suggestion, or ask something needing no memory). A simulated ASSISTANT from a different
model family replies; when the plan says the assistant suggests something at that turn,
it is told to. Each turn is verified — every planted value due must appear in its side's
text (normalised containment; numbers and names are planted to be checkable) — and
regenerated up to MAX_TRIES; a turn that never verifies is recorded, never silently kept.

Durable and resumable: one JSON line per finished turn in <out>/turns.jsonl.
Output: <out>/originals.json in the Z2 originals format (turn_number, timestamp, prompt,
response, timestamp_provenance) for experiments/z2/replay.py.

    uv run --no-sync python experiments/synth/generate.py --plan S1 --out logs/final-synth/S1
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.synth.plans import PLANS, STYLES, validate  # noqa: E402

POLICY = "ice-final-synth-generate-2"           # 2: cloud sides, invented-value checks (2026-10-28)
RECENT_PAIRS, MAX_TRIES = 5, 5        # the simulated user sees the plan + the last 5 exchanges; 5 tries
                                       # (was 3: cheap on the cloud, and a paraphrased value is rerolled)
ASSISTANT_RECENT_PAIRS = 8             # the simulated assistant: a plan-built briefing + the last 8
                                       # exchanges (FINAL_experiments.md §2.1, decided 2026-10-26)
SESSION_START, TURN_GAP = "19:00", timedelta(minutes=3)
# Each side is "local:<ollama model>" or "cloud:<Z2 profile role>". FINAL (maintainer, 2026-10-28):
# the two writer models on the cloud, plain text — MiMo-V2.6-Flash writes the user (more turns
# verified first time and terser, more natural messages than DeepSeek's in the speaker trial,
# experiments/synth/speaker_trial.py), DeepSeek V4.1 Flash the assistant. Was local Mistral-family
# models (ministral-3:8b / mistral-nemo, ~27 GPU hours for all nine).
USER_SIDE, ASSISTANT_SIDE = "cloud:mimo_chat", "cloud:deepseek_chat"
LOCAL_NUM_CTX = 32_768

USER_SYSTEM = """You write the next message of a USER chatting with an AI assistant, as the person
described in PERSONA, in exactly the writing style of STYLE. Stay in character; never mention
being simulated. Follow SESSION_BEATS for what this person wants today. The message MUST
contain every item in MUST_SAY, worded naturally but with each VALUE written exactly as given
(numbers, names and dates verbatim). An item marked "paste" goes inside pasted material
(a config, log, document or code block) that the user pastes, and its VALUE appears there word
for word (a comment or a line of the paste). An item marked "reject" means
the user turns down that earlier suggestion. If NO_MEMORY_TOPIC is set, the message is
instead a self-contained question about it. Never state a specific value (a number, amount,
date, time, name or place) that is not in MUST_SAY, PERSONA or RECENT: where SESSION_BEATS
hints at one, talk around it until MUST_SAY gives it, and never correct or change a value unless
MUST_SAY marks it a change. Write it the way a real person types into a chat
box, never the way an AI writes: no headings, bullet lists, bold or numbered steps unless STYLE
calls for structure or the text is pasted material; no em dashes; no stock assistant phrases
("Certainly", "Great question", "I hope this helps"); no summarising what was already said; no
thanking the assistant every message. Length, typos and slang follow STYLE. Return only the
message text."""

ASSISTANT_SYSTEM = """You are a helpful AI assistant in an ongoing conversation. Reply to the user's latest
message naturally and usefully. WHAT YOU KNOW lists what has been established earlier in this
conversation: use it to stay consistent, but bring a point up only when the user's message is
about it — do not recap it. Never invent facts about the user beyond what you know and the
conversation shown."""

ATTRIBUTION = {"decided": "the user decided", "preference": "the user's preference",
               "instruction": "the user's standing instruction to you", "hypothetical": "the user only mused about",
               "suggested": "you suggested", "rejected": "you suggested and the user turned down"}


def session_index(plan):
    out, turn = {}, 0
    for s in plan["sessions"]:
        for _ in range(s["pairs"]):
            turn += 1
            out[turn] = s["index"]
    return out


def never_again(fact):
    """Planted once and never touched again (S4's retention facts)."""
    return len(fact["values"]) == 1 and not fact.get("mentions")


def briefing(plan, turn, sessions=None):
    """The LOCAL facts the assistant may know before *turn*, built from the plan only: already
    said, current value, attributed, active in this or the previous session or due again in this
    one; never-again facts drop out once said; no probe ever shapes it."""
    sessions = sessions or session_index(plan)
    now = sessions[turn]
    lines = []
    for f in plan["facts"]:
        touches = sorted([v["turn"] for v in f["values"]] + list(f.get("mentions") or []))
        past = [t for t in touches if t < turn]
        if not past or never_again(f):
            continue
        future = [t for t in touches if t >= turn]
        active = sessions[past[-1]] >= now - 1 or (future and sessions[future[0]] == now)
        if active:
            lines.append(f"- {f['about']}: {value_before(f, turn)} ({ATTRIBUTION[f['status']]})")
    for e in plan.get("events") or []:
        if e["told_turn"] < turn and sessions[e["told_turn"]] >= now - 1:
            lines.append(f"- {e['what']} (on {e['date']})")
    return "\n".join(lines)


def recent_messages(history, user, n=ASSISTANT_RECENT_PAIRS):
    msgs = []
    for pair in history[-n:]:
        msgs += [dict(role="user", content=pair["user"]), dict(role="assistant", content=pair["assistant"])]
    return msgs + [dict(role="user", content=user)]


def drift_problems(plan, turn, user, reply):
    """A briefed fact restated with an outdated value by the assistant, or a never-again fact
    reappearing on either side after it was said."""
    out = []
    for f in plan["facts"]:
        said = [v for v in f["values"] if v["turn"] < turn]
        if not said:
            continue
        if never_again(f) and len(norm(f["values"][0]["value"])) >= 4:
            if contains(user, f["values"][0]["value"]) or contains(reply, f["values"][0]["value"]):
                out.append(f"never-again fact {f['id']} reappeared")
            continue
        current = norm(said[-1]["value"])
        for old in said[:-1]:
            o = norm(old["value"])
            if len(o) >= 4 and o not in current and contains(reply, old["value"]):
                out.append(f"assistant used outdated value of {f['id']}: {old['value']}")
    return out


from experiments.lme.cloud_provider import CloudCompletionError  # noqa: E402

# A reply cut at the output cap (cloud, or local_caller's RuntimeError) is one failed attempt;
# quota and key errors still stop the run.
CUT_OFF = (CloudCompletionError, RuntimeError)
NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
FENCE = re.compile(r"```.*?(?:```|$)", re.S)


def numbers(text):
    return {n.replace(",", "") for n in NUMBER.findall(text or "")}


def invented_values(plan, turn, user, history, allowed_text):
    """The user side may only say what the plan has given it (maintainer, 2026-10-28 trial: both
    writer models put an offer salary in the user's mouth turns before the plan's £58,000):
    a planned value before its turn, or a number outside pasted material that neither the plan's
    items for this turn, the persona, the beats nor the conversation so far contains."""
    out, said = [], " ".join(f"{p['user']} {p['assistant']}" for p in history)
    for f in plan["facts"]:
        for v in f["values"]:
            if v["turn"] > turn and (any(c.isdigit() for c in v["value"]) or len(v["value"].split()) >= 2) \
                    and contains(user, v["value"]) and not contains(said, v["value"]):
                out.append(f"user said {f['id']} early: {v['value']}")
    allowed = numbers(allowed_text) | numbers(said)
    if any(key in user for key in ('"PERSONA"', '"MUST_SAY"', '"SESSION_BEATS"')):
        return ["user side echoed its instructions"]
    loose = sorted(numbers(FENCE.sub(" ", user)) - allowed)
    if loose:
        out.append(f"user stated values the plan never gave: {', '.join(loose)}")
    return out


def hide_future(plan, turn, beats):
    """The beats (or the persona) with every planned value not yet due replaced by what it is about: neither side
    can say early what it never sees (2026-10-28 trial: a game's name in the beats was said
    turns before its plan turn by both writer models)."""
    def hide(text):
        for f in plan["facts"]:
            for v in f["values"]:
                if v["turn"] >= turn:
                    words = [re.escape(w) for w in re.split(r"[\s\-_,.]+", v["value"]) if w]
                    if words:
                        text = re.sub(r"[\s\-_,.]+".join(words), f"[{f['about']}, not said yet]", text, flags=re.I)
        return text
    if isinstance(beats, dict):                        # the persona: every text field
        return {k: hide(v) if isinstance(v, str) else v for k, v in beats.items()}
    return [hide(beat) for beat in beats]


def suggest_note(items):
    """This turn's planned assistant suggestions, as an instruction appended to the system prompt."""
    if not items:
        return ""
    lines = "\n".join(f"- {i['about']}: {i['value']}" for i in items)
    return ("\n\nThis reply MUST also propose each of the following as a suggestion (not as something already "
            "decided), even if the user's message is about something else, writing each value exactly as given:"
            f"\n{lines}")





def norm(text):
    return re.sub(r"[\s\-_,.]+", " ", (text or "").lower()).strip()


def contains(text, value):
    return norm(value) in norm(text)


def due(plan):
    """turn -> {"user": [...], "assistant": [...], "quiet": bool}."""
    out = {}
    slot = lambda t: out.setdefault(t, dict(user=[], assistant=[], quiet=False))
    for f in plan["facts"]:
        for v in f["values"]:
            side = "assistant" if f["speaker"] == "assistant" else "user"
            item = dict(id=f["id"], about=f["about"], value=v["value"], change=v["change"],
                        status=f["status"], paste=bool(f.get("in_paste")))
            slot(v["turn"])[side].append(item)
        mentions = list(f.get("mentions") or [])
        if f["status"] == "rejected":
            # The user turns the suggestion down at its first mention (plans record the
            # rejection there), or on the next turn when the plan gives none.
            turn = mentions.pop(0) if mentions else f["values"][0]["turn"] + 1
            slot(turn)["user"].append(dict(id=f["id"], about=f["about"], value=f["values"][0]["value"],
                                           change="reject", status="rejected", paste=False))
        for m in mentions:
            slot(m)["user"].append(dict(id=f["id"], about=f["about"], value=value_before(f, m),
                                        change="mention", status=f["status"], paste=False))
    for e in plan.get("events") or []:
        slot(e["told_turn"])["user"].append(dict(id=e["id"], about=e["what"], value=e["relative_phrase"],
                                                 change="event", status="event", paste=False))
    for t in plan.get("no_memory_turns") or []:
        slot(t)["quiet"] = True
    return out


def value_before(fact, turn):
    said = [v for v in fact["values"] if v["turn"] < turn]
    return said[-1]["value"] if said else fact["values"][0]["value"]


def session_of(plan):
    out, turn, rows = {}, 0, []
    for s in plan["sessions"]:
        start = datetime.fromisoformat(f"{s['date']}T{SESSION_START}")
        for k in range(s["pairs"]):
            turn += 1
            out[turn] = (s, start + k * TURN_GAP)
    return out


def generate_turn(plan, turn, session, history, items, *, user_llm, assistant_llm, sessions=None):
    """One verified pair, or the last attempt with its problems."""
    recent = history[-RECENT_PAIRS:]
    must_say = [dict(about=i["about"], value=i["value"], kind=("paste" if i["paste"] else i["change"]))
                for i in items["user"]]
    problems, user, reply = [], "", ""
    beats = hide_future(plan, turn, session["beats"])
    for attempt in range(1, MAX_TRIES + 1):
        try:
            user = user_llm(USER_SYSTEM, json.dumps(dict(
                PERSONA=hide_future(plan, turn, plan["persona"]), STYLE=STYLES[session["style"]], SESSION_BEATS=beats,
                RECENT=recent, MUST_SAY=must_say,
                NO_MEMORY_TOPIC=(session["topics"][0] if items["quiet"] and not must_say else None)),
                ensure_ascii=False))
            known = briefing(plan, turn, sessions)
            # a planned suggestion leads and closes the system prompt: at the end alone, both
            # writer models answered the user's topic and left it out (2026-10-28 trial)
            system = (suggest_note(items["assistant"]).strip() + "\n\n" if items["assistant"] else "") + (
                      ASSISTANT_SYSTEM + f"\n\nWHAT YOU KNOW:\n{known or '- nothing yet'}"
                      + "\n\nTODAY THE USER WANTS:\n" + "\n".join(f"- {b}" for b in beats)
                      + suggest_note(items["assistant"]))
            reply = assistant_llm(system, recent_messages(history, user))
        except CUT_OFF as exc:                        # a reply cut at the output cap: try again
            problems = [f"reply cut off: {str(exc)[:120]}"]
            continue
        problems = ([f"user missing {i['id']}: {i['value']}" for i in items["user"]
                     if i["change"] != "event" and not contains(user, i["value"])]
                    + [f"assistant missing {i['id']}: {i['value']}" for i in items["assistant"]
                       if not contains(reply, i["value"])]
                    + drift_problems(plan, turn, user, reply)
                    + invented_values(plan, turn, user, history, json.dumps(
                        dict(must=must_say, persona=plan["persona"], beats=session["beats"]), ensure_ascii=False)))
        if not problems:
            return dict(turn=turn, user=user, assistant=reply, attempts=attempt, verified=True)
    return dict(turn=turn, user=user, assistant=reply, attempts=MAX_TRIES, verified=False, problems=problems)


def run(plan, out, *, user_llm, assistant_llm, say=print, limit=None):
    """Generates every turn not yet in <out>/turns.jsonl; *limit* stops after that turn (a
    trial: no originals.json, since the conversation is unfinished)."""
    if validate(plan):
        raise ValueError(f"{plan['id']}: plan does not validate")
    out.mkdir(parents=True, exist_ok=True)
    store = out / "turns.jsonl"
    done = {}
    if store.exists():
        for line in store.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["turn"]] = row
    schedule, when, sessions = due(plan), session_of(plan), session_index(plan)
    history = [dict(user=done[t]["user"], assistant=done[t]["assistant"]) for t in sorted(done)]
    total = len(when)
    for turn in range(1, total + 1):
        if limit and turn > limit:
            return dict(plan=plan["id"], turns=limit, unverified=[t for t in done if not done[t]["verified"]],
                        finished=False)
        if turn in done:
            continue
        session, _ = when[turn]
        items = schedule.get(turn, dict(user=[], assistant=[], quiet=False))
        row = generate_turn(plan, turn, session, history, items, user_llm=user_llm, assistant_llm=assistant_llm,
                            sessions=sessions)
        with store.open("a") as handle:
            handle.write(json.dumps(dict(row, policy=POLICY), ensure_ascii=False) + "\n")
        done[turn] = row
        history.append(dict(user=row["user"], assistant=row["assistant"]))
        if not row["verified"]:
            say(f"WARNING {plan['id']} turn {turn}: unverified after {MAX_TRIES} tries: {row['problems']}")
    rows = [dict(turn_number=t, timestamp=when[t][1].isoformat() + "Z", timestamp_provenance="synthetic_plan",
                 prompt=done[t]["user"], response=done[t]["assistant"]) for t in range(1, total + 1)]
    (out / "originals.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    unverified = [t for t in done if not done[t]["verified"]]
    status = dict(plan=plan["id"], turns=total, unverified=unverified, finished=True)
    (out / "status.json").write_text(json.dumps(status, indent=1) + "\n")
    return status


def local_caller(model, num_ctx=LOCAL_NUM_CTX):
    """(system, prompt or chat messages) -> text on a local Ollama model (thinking off, held
    while generating; Ollama reuses its cache for a prompt that extends the previous one)."""
    import urllib.request
    from src.api.config import settings

    def call(system, prompt):
        turns = prompt if isinstance(prompt, list) else [dict(role="user", content=prompt)]
        body = dict(model=model, stream=False, think=False, keep_alive="30m",
                    messages=[dict(role="system", content=system)] + turns,
                    options=dict(num_ctx=num_ctx, temperature=0.8, num_predict=2_000))
        request = urllib.request.Request(settings.ollama_base_url.rstrip("/") + "/api/chat",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            out = json.load(response)
        if out.get("done_reason") == "length":
            raise RuntimeError(f"{model}: reply cut at the output cap")
        return (out.get("message") or {}).get("content", "").strip()
    return call


def caller(side, *, session="", usage_log=None):
    kind, _, name = side.partition(":")
    if kind == "local":
        return local_caller(name)
    if kind == "cloud":
        return cloud_caller(name, session=session, usage_log=usage_log)
    raise ValueError(f"side must be local:<model> or cloud:<role>, got {side!r}")


def cloud_caller(role, *, session="", usage_log=None):
    """(system, prompt) -> text on a Z2 profile (cloud; sized from measured cost first, TRAPS 148).
    One provider session per conversation and side, so the fixed system text is served from the
    cache; each call's usage (cost included) is appended to *usage_log*."""
    import time
    from experiments.z2.profiles import generator
    gen = generator(role)

    def call(system, prompt):
        turns = prompt if isinstance(prompt, list) else [dict(role="user", content=prompt)]
        started = time.time()
        result = gen.generate([dict(role="system", content=system)] + turns,
                              temperature=0.8, max_output_tokens=4_000, session_id=f"ice-final-synth-{session}{role}")
        if usage_log:
            with Path(usage_log).open("a") as handle:
                handle.write(json.dumps(dict(role=role, seconds=round(time.time() - started, 2),
                                             usage=result.usage)) + "\n")
        return result.text
    return call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--user", default=USER_SIDE, help="local:<ollama model> | cloud:<role>")
    parser.add_argument("--assistant", default=ASSISTANT_SIDE, help="local:<ollama model> | cloud:<role>")
    parser.add_argument("--limit", type=int, help="stop after this turn (a trial)")
    args = parser.parse_args()
    plan = json.loads((PLANS / f"{args.plan}.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    sides = dict(user=args.user, assistant=args.assistant, policy=POLICY)
    sides_path = args.out / "sides.json"
    if (args.out / "turns.jsonl").exists() and sides_path.exists():
        before = json.loads(sides_path.read_text())
        if (before["user"], before["assistant"]) != (args.user, args.assistant):
            raise SystemExit(f"{args.out} was generated by {before['user']} / {before['assistant']}: "
                             "one conversation never mixes models — move it aside first")
    sides_path.write_text(json.dumps(sides) + "\n")
    usage = args.out / "usage.jsonl"
    status = run(plan, args.out, limit=args.limit,
                 user_llm=caller(args.user, session=f"{args.plan}-user-", usage_log=usage),
                 assistant_llm=caller(args.assistant, session=f"{args.plan}-assistant-", usage_log=usage))
    print(json.dumps(status))
    return 0 if not status["unverified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
