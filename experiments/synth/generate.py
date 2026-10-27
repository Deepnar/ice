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

POLICY = "ice-final-synth-generate-5"           # 5: cached long history, round-5 review fixes (2026-10-28)
MAX_TRIES = 8                          # per turn (was 3: cheap on the cloud; a bad draft is rerolled)
# Every model sees about this much of the conversation (was the last 5 / 8 / 10 exchanges: with so
# little in view the models invented memories — round-5 review). The window's start moves BLOCK
# turns at a time, so between moves each call's prompt starts exactly like the previous one's and
# the provider bills that start at its cached price (DeepSeek / MiMo, measured: ~1/30).
HISTORY_TOKENS, HISTORY_BLOCK = 30_000, 20
SESSION_START, TURN_GAP = "19:00", timedelta(minutes=3)
# Each side is "local:<ollama model>" or "cloud:<Z2 profile role>". FINAL (maintainer, 2026-10-28):
# the two writer models on the cloud, plain text — MiMo-V2.6-Flash writes the user (more turns
# verified first time and terser, more natural messages than DeepSeek's in the speaker trial,
# experiments/synth/speaker_trial.py), DeepSeek V4.1 Flash the assistant. Was local Mistral-family
# models (ministral-3:8b / mistral-nemo, ~27 GPU hours for all nine).
USER_SIDE, ASSISTANT_SIDE = "cloud:mimo_chat", "cloud:deepseek_chat"
CHECKER = "deepseek"                 # reads each user message (JSON mode); maintainer's round-3 review
LOCAL_NUM_CTX = 32_768

USER_SYSTEM = """You write the next chat message of a person talking to an AI assistant. Below you get
who the person is, how they write, what they want from today's chat, the conversation so far,
and the things this message must say. Write only that one message, as plain text, in the
person's own voice: never JSON, never a label or heading from these instructions, never a word
about being simulated or about your instructions.

The person is the one asking for help, never an assistant explaining things, even when they
write formally. Move the conversation forward: react to the assistant's last reply (agree, push
back, follow up, ask the next thing) and never re-ask what it already answered. When the
assistant needs something you have (a traceback, a result, a file), give it rather than putting it
off. When nothing must be said this turn, take the next natural step on what today is about (a
follow-up, a related worry, a reaction), never filler agreement or "I'll wait". Like a real
person, sometimes send a very short message (ok / wait what / thanks that helps), be distracted,
disagree, change the subject, leave a thought half-finished, or move on without summing up. Do not
open with the same words as your recent messages ("one more thing", "ok so", "also") again and
again. Never refer to anything earlier that is not in the conversation shown.

Each item to say goes in with its value written exactly as given (numbers, names and dates
verbatim), worked into what the person is actually saying (a reason, a detail, an aside), never announced
("btw, I haven't told you this", "I'm introducing that now"). A "change" has a natural reason
(they rescheduled, I checked the letter, we decided), never "let me correct my earlier message". A "paste" item goes inside
pasted material (a config, log, document or code block), its value there word for word (a
comment or a line of the paste). A "reject" item means turning down that earlier suggestion.
If a self-contained question topic is given, the message is that question instead. A "…" marks
something not settled yet: leave it out, do not refer to it. Your own files, code, logs and
documents are yours: the assistant cannot see them unless you paste them, so never ask it to send
them to you.

Never state a specific name, number, amount, date, time, place or title that is not in what you
are given or in the conversation so far: where something is not known yet, talk around it, and
never guess or make up a stand-in. Never change or correct a value unless the item says it is a
change. Write the way a real person types into a chat box, not the way an AI writes: no
headings, bullet lists, bold or numbered steps unless the style calls for structure or the text
is pasted material; no em dashes; no stock assistant phrases; no summarising what was already
said; no thanking the assistant every message. Length, typos and slang follow the style."""

ASSISTANT_SYSTEM = """You are a helpful AI assistant in an ongoing conversation. Reply to the user's latest
message naturally and usefully. WHAT YOU KNOW lists what has been established earlier in this
conversation: use it to stay consistent, but bring a point up only when the user's message is
about it — do not recap it. Never invent facts about the user beyond what you know and the
conversation shown; never mention anything the user has not said in this conversation. Match
the user's message: a quick message gets a short reply (one to four sentences); use lists or
headings only when the content really is a list; do not end every reply with a question or a
summary. You can only chat: you cannot set reminders, schedule, send messages, browse or act
outside this conversation, so never offer to. Do not just agree: say so when something does not
add up, and do not restate what was already said."""


def window(history):
    """The exchanges every model sees: the newest ~HISTORY_TOKENS, starting on a HISTORY_BLOCK
    boundary — the start only moves forward, a block at a time, so the prompt start is stable."""
    sizes = [(len(p["user"]) + len(p["assistant"])) // 4 for p in history]
    start, total = 0, sum(sizes)
    while start < len(history) and total > HISTORY_TOKENS:
        total -= sum(sizes[start:start + HISTORY_BLOCK])
        start += HISTORY_BLOCK
    return history[start:]


def transcript(pairs, me="User"):
    """A window as text, the same bytes every time it is shown (the cached prompt start)."""
    return "\n\n".join(f"{who}: {text}" for p in pairs for who, text in ((me, p["user"]), ("Assistant", p["assistant"])))


def user_prompt(persona, style, beats, beat_now, recent, must_say, topic, last=False, invite=(), unsettled=()):
    """The simulated user's input as plain labelled text (given as JSON, MiMo sometimes answered in
    JSON too — round-3 sample): the conversation first, the same from turn to turn so the provider
    caches it, then this turn's notes."""
    lines = ["THE CONVERSATION SO FAR (latest last):", transcript(recent, me="You") or
             "(nothing yet: this is the first message today)", "", "---", "WHO YOU ARE:"]
    lines += [f"{k}: {v}" for k, v in persona.items()]
    lines += ["", f"HOW YOU WRITE: {style}", "", "WHAT TODAY'S CHAT IS ABOUT:"] + [f"- {b}" for b in beats]
    if beat_now:
        lines += ["", f"WHERE IT IS NOW: {beat_now}"]
    if unsettled:                                   # round 6: MiMo kept filling these in with guesses
        lines += ["", "NOT SETTLED YET (you do not know these yet, so give no value for them):"]
        lines += [f"- {u}" for u in unsettled]
    lines += ["", "THIS MESSAGE MUST SAY:"]
    lines += [f"- {i['about']}: {i['value']} ({i['kind']})" for i in must_say] or ["(nothing specific this turn)"]
    if invite:
        lines += ["", "ALSO ASK THE ASSISTANT FOR ITS IDEAS ON: " + "; ".join(invite)]
    if topic:
        lines += ["", f"INSTEAD, ASK A SELF-CONTAINED QUESTION ABOUT: {topic}"]
    lines += ["", "This is your last message in today's chat." if last else
              "Today's chat goes on after this message: no goodbyes or wrapping up."]
    return "\n".join(lines) + "\n\nWrite the message now."

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


def rejected_turn(fact):
    """When the user turns a rejected suggestion down: its first mention, else the turn after (due())."""
    return (fact.get("mentions") or [fact["values"][0]["turn"] + 1])[0]


def attribution(fact, turn):
    """Who said it, as of *turn*: a suggestion is "turned down" only once the user has turned it
    down (round-5 review: the assistant called one rejected before the rejection)."""
    if fact["status"] == "rejected" and rejected_turn(fact) >= turn:
        return ATTRIBUTION["suggested"]
    return ATTRIBUTION[fact["status"]]


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
            lines.append(f"- {f['about']}: {value_before(f, turn)} ({attribution(f, turn)})")
    for e in plan.get("events") or []:
        if e["told_turn"] < turn and sessions[e["told_turn"]] >= now - 1:
            lines.append(f"- {e['what']} (on {e['date']})")
    return "\n".join(lines)


def assistant_messages(history, notes, user):
    """The assistant's conversation: the window as chat turns (the cached start), this turn's notes
    as a system message after it, then the user's new message."""
    msgs = []
    for pair in window(history):
        msgs += [dict(role="user", content=pair["user"]), dict(role="assistant", content=pair["assistant"])]
    return msgs + [dict(role="system", content=notes), dict(role="user", content=user)]


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
# A reply in JSON, or one naming its instructions (round-3 sample: {"messages": [...]} replies and
# "MUST_SAY is empty"), is never a chat message.
# An unfinished placeholder left in a message ("[some weeks]" in a reply, round-5 review).
ARTIFACT = re.compile(r"\[[a-z][a-z /'-]{1,40}\](?!\()|\{\{|<insert|\bTODO\b|\bTBD\b|\bPLACEHOLDER\b")


def artifacts(text):
    found = ARTIFACT.findall(FENCE.sub(" ", text or ""))
    return [f"unfinished placeholder: {found[0]}"] if found else []


LEAK = re.compile(r'^\s*(\{\s*"|\[\s*\{)|MUST_SAY|SESSION_BEATS|BEAT_NOW|NO_MEMORY_TOPIC|"PERSONA"|WHO YOU ARE:'
                  r"|HOW YOU WRITE:|WHERE IT IS NOW:|THIS MESSAGE MUST SAY|my instructions|as an AI\b")
FENCE = re.compile(r"```.*?(?:```|$)", re.S)


def numbers(text):
    return {n.replace(",", "") for n in NUMBER.findall(text or "")}


def invented_values(plan, turn, user, history):
    """The mechanical checks on a user message: a changing fact's later value before its turn
    (it would undo a planned change); a reply in JSON or naming its instructions; a repeat.
    Invented details are the checker's: only one that gives a value to a topic the plan has not
    settled yet, or contradicts what was said, is wrong (maintainer's round-4 review: a realistic
    detail like a rent figure is not; the old any-unplanned-number rule rejected it)."""
    out, earlier = [], " ".join(f"{p['user']} {p['assistant']}" for p in history)
    # A first value said early is how real people talk (maintainer's round-4 review); a LATER value
    # said before its turn would undo a planned change, so that alone is too early.
    for f in plan["facts"]:
        # a later value equal to one already due is not early (batch 6: rent £1,150 at turn 34 —
        # due then — is also its value again at turn 121; office days go 2 -> 3 -> 2)
        due_now = {norm(v["value"]) for v in f["values"] if v["turn"] <= turn}
        for v in f["values"][1:]:
            if v["turn"] > turn and norm(v["value"]) not in due_now and (any(c.isdigit() for c in v["value"]) or len(v["value"].split()) >= 2) \
                    and said(user, v["value"]) and not said(earlier, v["value"]):
                out.append(f"user said {f['id']}'s later value early: {v['value']}")
    if LEAK.search(user):
        return ["user side echoed its instructions"]
    out += artifacts(user)
    if any(norm(user)[:120] == norm(p["user"])[:120] for p in history):
        return ["user side repeated an earlier message"]
    return out


def hide_future(plan, turn, beats):
    """The beats (or the persona) with every planned value not yet due replaced by "…" (a bracketed
    "not said yet" made the user act it out; "the destination city" was copied word for word —
    maintainer's round-4 review), its numbers too: neither side
    can say early what it never sees (2026-10-28 trial: a game's name in the beats was said
    turns before its plan turn by both writer models)."""
    def hide(text):
        for f in plan["facts"]:
            for v in f["values"]:
                if v["turn"] >= turn:
                    words = [re.escape(w) for w in re.split(r"[\s\-_,.]+", v["value"]) if w]
                    if words:
                        text = re.sub(r"[\s\-_,.]+".join(words), "…", text, flags=re.I)
                    # its numbers too, wherever they stand alone (round 4: a persona's "2,400 USD"
                    # budget was said turns before the plan's "2400 usd all in, flights included")
                    for n in (x for x in NUMBER.findall(v["value"]) if len(x.replace(",", "")) >= 2):
                        text = re.sub(rf"(?<![\d.,]){re.escape(n)}(?![\d.,])", "…", text)
                        if "," not in n and len(n) >= 4:          # 2400 also as 2,400
                            text = re.sub(rf"(?<![\d.,]){n[:-3]},{n[-3:]}(?![\d.,])", "…", text)
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


# Words that carry no fact of their own: a long value's first clause is checked without them.
COMMON = set("""a an the and or but of to in on at for with by from up down out over about into as is are was
were be been being has have had do does did it its this that these those she he her his him they them their
i me my we our you your which who whom whose what when where why how not no so very too also just only still than
then there here likes like dislikes dislike prefers prefer hates hate loves love wants want uses use gets get got
gives give makes make picks pick takes take moved move goes go went lives live works work keeps keep thinks think
says say always never usually every each per about around roughly approx at assistant suggested suggests user rejected""".split())


def key_parts(value):
    """What a question about a long value needs (maintainer, 2026-10-28: a long value said in other
    words was a quarter of the unverified turns): its numbers, its capitalised words, and the
    content words of its first clause — "Ostmere, a lake town up north; she moved away at age 11"
    needs Ostmere and 11; "dislikes coriander, picks it out of everything" needs coriander."""
    first = re.split(r"[,;]", value, maxsplit=1)[0]
    parts = {w for w in re.findall(r"[A-Za-z][\w'\-]*|\d+(?:[.,:]\d+)*", value) if w[0].isdigit() or w[0].isupper()}
    return parts | {w for w in re.findall(r"[A-Za-z][\w'\-]*", first) if w.lower() not in COMMON and len(w) > 2}


def said(text, value):
    """The value word for word; a value of four words or more may be said in other words, as long
    as every key part appears as a whole word."""
    if contains(text, value):
        return True
    parts = key_parts(value) if len(value.split()) >= 4 else ()
    return bool(parts) and all(re.search(rf"(?<!\w){re.escape(norm(p))}(?!\w)", norm(text)) for p in parts)


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


CHECK_SYSTEM = """You check one chat message written by a simulated USER for a dataset. You get what is
ESTABLISHED about the person (their profile and the facts said so far), the CONVERSATION so far,
the topics NOT SETTLED YET, what THIS MESSAGE HAD TO SAY, and the MESSAGE. Report a problem only
when you are sure, and quote the exact words of the message it is about (for "missing", quote the
item). Ordinary details people mention (a rent figure, a friend's name, asking for a reminder or
for help with anything) are fine. Problems:
- "invented": the message gives a concrete value (a name, number, amount, date, time, place or
  title) for a topic NOT SETTLED YET, other than what it had to say; talking about a topic, asking
  about it or bringing it up without giving its value is fine;
- "contradiction": the message states something that conflicts with what is established or said
  earlier, other than a change it had to say;
- "not_a_message": JSON, labels, talk about instructions or being simulated, or writing as the
  assistant rather than the person asking;
- "repeat": it repeats an earlier user message in substance;
- "forgets": it treats as unknown, or asks again about, something already established;
- "contrived": only meta wording about its own message ("I'm introducing", "let me correct my
  earlier message", "for the record") or stalling instead of giving what the assistant asked for;
  a plain statement of a fact, however casual ("ok so the start date's monday"), is fine;
- "missing": an item listed under CHECK THESE ITEMS is not clearly conveyed. Same meaning counts:
  its names and numbers must be there but in any format ("10 April 2025, 9:10" for "thursday 10
  april 2025 at 9:10am"); a "reject" item only needs that suggestion clearly turned down; a
  "mention" item only needs it referred to with its value.
"contradiction" includes continuity: a decision, a split, what a label or number refers to, where
the person lives or is moving, who has which file — anything that silently changes from earlier.
"not_a_message" includes role reversal: explaining like an assistant, or asking the assistant for
something only the person has (their own code, logs, files). Asking the assistant for help,
advice, a plan or a reminder is always fine.
Return JSON: {"problems": [{"kind": "...", "quote": "...", "detail": "...", "is_problem": true}]} — an
empty list when there are none; if, on reflection, something is fine, leave it out or set
"is_problem": false."""

REPLY_CHECK = """You check one ASSISTANT reply in a chat for a dataset. You get the CONVERSATION so far,
what is ESTABLISHED (the facts said so far and who said them), the USER's latest message and the
REPLY. Report a problem only when you are sure, and quote the exact words of the reply it is about.
Problems:
- "off": the reply does not answer or follow the user's latest message;
- "contradiction": it conflicts with what was said earlier, or claims the user said something
  they did not;
- "invented_memory": it refers to an earlier exchange, suggestion or decision that is in neither
  the conversation nor what is established;
- "forgets": it treats as unknown something already established;
- "wrong": a clear factual or technical error (a wrong encoding fact, wrong arithmetic);
- "inconsistent": it contradicts the assistant's own earlier advice without saying it changed its mind;
- "impossible": it offers to do anything outside the chat (reminders, scheduling, sending,
  browsing) or to look at something the user has not pasted.
Return JSON: {"problems": [{"kind": "...", "quote": "...", "detail": "...", "is_problem": true}]} — an
empty list when there are none; if, on reflection, something is fine, leave it out or set
"is_problem": false."""


def upcoming(plan, turn):
    """The topics the plan settles later — what they are about, never their values."""
    return [re.sub(re.escape(f["values"][0]["value"]), "", f["about"], flags=re.I).strip(" ,'\"") or f["about"]
            for f in plan["facts"] if f["values"][0]["turn"] > turn and f["speaker"] != "assistant"]


def established(plan, turn, persona, suggestions=False):
    """What was said before *turn* (current values) and who the person is, for the checkers; with
    *suggestions*, the assistant's planned suggestions too, attributed as of the turn."""
    lines = [f"{k}: {v}" for k, v in persona.items()]
    for f in plan["facts"]:
        said_before = [v for v in f["values"] if v["turn"] < turn]
        if said_before and (f["speaker"] != "assistant" or suggestions):
            lines.append(f"- {f['about']}: {said_before[-1]['value']} ({attribution(f, turn)})")
    for e in plan.get("events") or []:
        if e["told_turn"] < turn:
            lines.append(f"- {e['what']} ({e['relative_phrase']})")
    return "\n".join(lines)


def checker_caller(role, *, session="", usage_log=None):
    """check(established, recent pairs, must_say, message, unconfirmed=(), reply=None) -> problems,
    from a cloud writer in JSON mode (maintainer's reviews, 2026-10-28: exact-string checks passed
    invented names, JSON replies and broken character, and failed items said in other words).
    With *reply*, it reads the assistant's reply instead. An unreadable verdict is a failed
    attempt, never a pass."""
    asks = {"user": cloud_caller(role, session=f"{session}user-", usage_log=usage_log),
            "reply": cloud_caller(role, session=f"{session}reply-", usage_log=usage_log)}

    def verdict(which, system, prompt, text, items=()):
        try:
            found = json.loads(asks[which](system, prompt))["problems"]
        except (ValueError, KeyError, TypeError) as exc:
            return [f"checker reply unreadable: {type(exc).__name__}"]
        # a problem counts only with a quote that is really there — in the text, or, for "missing",
        # the item (round 5: the checker flagged a message, then doubted itself in its own reason)
        real = lambda p: (p.get("kind") == "missing" and any(norm(str(p.get("quote")))[:30] in norm(f"{i['about']} {i['value']}")
                                                            for i in items)) \
            or (len(norm(str(p.get("quote") or ""))) >= 3 and norm(str(p.get("quote"))) in norm(text))
        # and only with an explicit yes (round 6: two "wrong" flags whose own reasons said "no error")
        return [f"checker {p.get('kind')}: \"{str(p.get('quote'))[:80]}\" {str(p.get('detail'))[:120]}"
                for p in found if isinstance(p, dict) and p.get("is_problem") is True and real(p)]

    def check(known, recent, must_say, message, unconfirmed=(), reply=None, upcoming=()):
        convo = f"CONVERSATION:\n{transcript(recent) or '(none yet)'}\n\n---\n"    # first: the cached start
        if reply is not None:
            return verdict("reply", REPLY_CHECK, f"{convo}ESTABLISHED:\n{known or '(nothing yet)'}\n\nUSER:\n{message}"
                                                 f"\n\nREPLY:\n{reply}", reply)
        items = "\n".join(f"- {i['about']}: {i['value']} ({i['kind']})" for i in must_say) or "(nothing specific)"
        doubt = "\n".join(f"- {i['about']}: {i['value']} ({i['kind']})" for i in unconfirmed) or "(none)"
        later = "\n".join(f"- {t}" for t in upcoming) or "(none)"
        return verdict("user", CHECK_SYSTEM, f"{convo}ESTABLISHED:\n{known or '(nothing yet)'}"
                                             f"\n\nNOT SETTLED YET:\n{later}\n\nTHIS MESSAGE HAD TO SAY:\n{items}\n\nCHECK THESE ITEMS:"
                                             f"\n{doubt}\n\nMESSAGE:\n{message}", message, unconfirmed)
    return check


def generate_turn(plan, turn, session, history, items, *, user_llm, assistant_llm, sessions=None, checker=None,
                  log=lambda record: None):
    """One verified pair, or the last attempt with its problems; every rejected attempt goes to
    *log* with its reasons (round-5 review: the checker's decisions must be visible)."""
    recent = window(history)
    must_say = [dict(about=i["about"], value=i["value"], kind=("paste" if i["paste"] else i["change"]))
                for i in items["user"]]
    problems, user, reply = [], "", ""
    beats = hide_future(plan, turn, session["beats"])
    persona = hide_future(plan, turn, plan["persona"])
    sessions = sessions or session_index(plan)
    first = min(t for t, s in sessions.items() if s == sessions[turn])
    # where the session should be about now: the beats paced over its pairs (maintainer's review,
    # 2026-10-28: with all beats at every turn the user kept re-asking the first one)
    beat_now = beats[min(len(beats) - 1, (turn - first) * len(beats) // max(1, session["pairs"]))] if beats else None
    topic = session["topics"][0] if items["quiet"] and not must_say else None
    last = turn == first + session["pairs"] - 1
    # a planned suggestion needs the user to open the door: asked about Dario's playtesting, both
    # writer models declined to drop in a company name (S2 turn 6, every round) — the user asks
    # about its topic, never its value
    invite = [re.sub(re.escape(i["value"]), "", i["about"], flags=re.I).strip(" ,'\"") or i["about"]
              for i in items["assistant"]]
    for attempt in range(1, MAX_TRIES + 1):
        try:
            user = user_llm(USER_SYSTEM, user_prompt(persona, STYLES[session["style"]], beats, beat_now, recent,
                                                     must_say, topic, last, invite,
                                                     [u for u in upcoming(plan, turn) if u not in {m["about"] for m in must_say}]))
            # the user message first: mechanical checks, then the checker's reading; a bad message
            # never costs an assistant reply. An item not found word for word goes to the checker,
            # which judges it by meaning (round-4 review: a date in another format, a rejection in
            # other words); without a checker it fails.
            unconfirmed = [m for i, m in zip(items["user"], must_say) if i["change"] == "event" or not said(user, i["value"])]
            problems = invented_values(plan, turn, user, history)
            if not checker:
                problems += [f"user missing: {m['value']}" for m, i in zip(must_say, items["user"])
                             if m in unconfirmed and i["change"] != "event"]
            elif not problems:
                problems = checker(established(plan, turn, persona), recent, must_say, user, unconfirmed,
                                   upcoming=upcoming(plan, turn))
            if problems:
                log(dict(turn=turn, attempt=attempt, stage="user", problems=problems, user=user))
                continue
            known = briefing(plan, turn, sessions)
            # a planned suggestion leads and closes the system prompt: at the end alone, both
            # writer models answered the user's topic and left it out (2026-10-28 trial)
            # No beats for the assistant (maintainer's review, 2026-10-28): from them it brought up
            # what the user had not said yet; it knows the conversation and the briefing only.
            # the system prompt stays fixed (the cached start); this turn's notes follow the window
            notes = ((suggest_note(items["assistant"]).strip() + "\n\n" if items["assistant"] else "")
                     + f"WHAT YOU KNOW:\n{known or '- nothing yet'}" + suggest_note(items["assistant"]))
            reply = assistant_llm(ASSISTANT_SYSTEM, assistant_messages(history, notes, user))
        except CUT_OFF as exc:                        # a reply cut at the output cap: try again
            problems = [f"reply cut off: {str(exc)[:120]}"]
            log(dict(turn=turn, attempt=attempt, stage="call", problems=problems))
            continue
        # a long planned suggestion by its key parts, like the user's items (batch 6: S6's value
        # "assistant suggested brown rice for Kit's dinners; rejected, he won't eat it")
        problems = ([f"assistant missing {i['id']}: {i['value']}" for i in items["assistant"]
                     if not said(reply, i["value"])]
                    + drift_problems(plan, turn, user, reply) + artifacts(reply))
        if not problems and checker:                  # follows, continuity, facts, stays in the chat
            try:
                problems = checker(established(plan, turn, persona, suggestions=True), recent, [], user, reply=reply)
            except CUT_OFF as exc:
                problems = [f"reply cut off: {str(exc)[:120]}"]
        if problems:
            log(dict(turn=turn, attempt=attempt, stage="reply", problems=problems, user=user, reply=reply))
        if not problems:
            return dict(turn=turn, user=user, assistant=reply, attempts=attempt, verified=True)
    return dict(turn=turn, user=user, assistant=reply, attempts=MAX_TRIES, verified=False, problems=problems)


def run(plan, out, *, user_llm, assistant_llm, say=print, limit=None, checker=None):
    """Generates every turn not yet in <out>/turns.jsonl; *limit* stops after that turn (a
    trial: no originals.json, since the conversation is unfinished). A turn that never verifies
    STOPS the conversation (maintainer's round-4 review: a failed turn left a blank reply the
    next turns talked past): it is written to <out>/stopped.json, never kept, and the run ends
    unfinished. Rerunning tries it again; <out>/repair-<turn>.json ({"user": ..., "assistant": ...},
    written by hand) is used as that turn."""
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

    def attempt_log(record):                          # every rejected attempt and why (attempts.jsonl)
        with (out / "attempts.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(record, policy=POLICY), ensure_ascii=False) + "\n")
    history = [dict(user=done[t]["user"], assistant=done[t]["assistant"]) for t in sorted(done)]
    total = len(when)
    for turn in range(1, total + 1):
        if limit and turn > limit:
            return dict(plan=plan["id"], turns=limit, finished=False)
        if turn in done:
            continue
        session, _ = when[turn]
        items = schedule.get(turn, dict(user=[], assistant=[], quiet=False))
        repair = out / f"repair-{turn}.json"
        if repair.exists():
            fixed = json.loads(repair.read_text())
            row = dict(turn=turn, user=fixed["user"], assistant=fixed["assistant"], attempts=0, verified=True, repaired=True)
        else:
            row = generate_turn(plan, turn, session, history, items, user_llm=user_llm, assistant_llm=assistant_llm,
                                sessions=sessions, checker=checker, log=attempt_log)
        if not row["verified"]:
            (out / "stopped.json").write_text(json.dumps(dict(row, policy=POLICY), indent=1, ensure_ascii=False) + "\n")
            say(f"STOPPED {plan['id']} at turn {turn}: no attempt verified in {MAX_TRIES}: {row['problems']} — "
                f"rerun to try again, or write {repair} by hand")
            return dict(plan=plan["id"], turns=total, stopped_at=turn, problems=row["problems"], finished=False)
        (out / "stopped.json").unlink(missing_ok=True)
        with store.open("a") as handle:
            handle.write(json.dumps(dict(row, policy=POLICY), ensure_ascii=False) + "\n")
        done[turn] = row
        history.append(dict(user=row["user"], assistant=row["assistant"]))
    rows = [dict(turn_number=t, timestamp=when[t][1].isoformat() + "Z", timestamp_provenance="synthetic_plan",
                 prompt=done[t]["user"], response=done[t]["assistant"]) for t in range(1, total + 1)]
    (out / "originals.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    status = dict(plan=plan["id"], turns=total, repaired=sorted(t for t in done if done[t].get("repaired")), finished=True)
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
                              temperature=0.8, max_output_tokens=8_000, session_id=f"ice-final-synth-{session}{role}")  # 8K: long technical replies were cut at 4K
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
    parser.add_argument("--checker", default=CHECKER, help="cloud writer role that reads each user message, or none")
    args = parser.parse_args()
    plan = json.loads((PLANS / f"{args.plan}.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    sides = dict(user=args.user, assistant=args.assistant, checker=args.checker, policy=POLICY)
    sides_path = args.out / "sides.json"
    if (args.out / "turns.jsonl").exists() and sides_path.exists():
        before = json.loads(sides_path.read_text())
        if (before["user"], before["assistant"]) != (args.user, args.assistant):
            raise SystemExit(f"{args.out} was generated by {before['user']} / {before['assistant']}: "
                             "one conversation never mixes models — move it aside first")
    sides_path.write_text(json.dumps(sides) + "\n")
    usage = args.out / "usage.jsonl"
    checker = None if args.checker == "none" else checker_caller(args.checker, session=f"{args.plan}-check-", usage_log=usage)
    status = run(plan, args.out, limit=args.limit, checker=checker,
                 user_llm=caller(args.user, session=f"{args.plan}-user-", usage_log=usage),
                 assistant_llm=caller(args.assistant, session=f"{args.plan}-assistant-", usage_log=usage))
    print(json.dumps(status))
    return 0 if status["finished"] or args.limit else 4               # 4: stopped at a turn for repair


if __name__ == "__main__":
    raise SystemExit(main())
