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

POLICY = "ice-final-synth-generate-1"
RECENT_PAIRS, MAX_TRIES = 5, 3        # the simulated user sees the plan + the last 5 exchanges
ASSISTANT_HISTORY_TOKENS = 28_000      # the simulated assistant sees the whole conversation, oldest
                                       # trimmed only past its window (as a real chatbot would)
SESSION_START, TURN_GAP = "19:00", timedelta(minutes=3)
# Each side is "local:<ollama model>" or "cloud:<Z2 profile role>". Default (maintainer,
# 2026-10-26): both sides local — the user from the plan, the last RECENT_PAIRS exchanges and
# the items due; the assistant from the whole conversation, like a real chatbot. Mistral-family
# models: neither Gemma (ICE's background model) nor Qwen (the answerer and embedder).
USER_SIDE, ASSISTANT_SIDE = "local:ministral-3:8b", "local:mistral-nemo"
LOCAL_NUM_CTX = 32_768

USER_SYSTEM = """You write the next message of a USER chatting with an AI assistant, as the person
described in PERSONA, in exactly the writing style of STYLE. Stay in character; never mention
being simulated. Follow SESSION_BEATS for what this person wants today. The message MUST
contain every item in MUST_SAY, worded naturally but with each VALUE written exactly as given
(numbers, names and dates verbatim). An item marked "paste" goes inside pasted material
(a config, log, document or code block) that the user pastes. An item marked "reject" means
the user turns down that earlier suggestion. If NO_MEMORY_TOPIC is set, the message is
instead a self-contained question about it. Return only the message text."""

ASSISTANT_SYSTEM = """You are a helpful AI assistant in an ongoing conversation. Reply to the user's latest
message naturally and usefully, drawing on everything said earlier in the conversation. Never
invent facts about the user beyond the conversation."""


def suggest_note(items):
    """This turn's planned assistant suggestions, as an instruction appended to the system prompt."""
    if not items:
        return ""
    lines = "\n".join(f"- {i['about']}: {i['value']}" for i in items)
    return ("\n\nIn this reply, propose each of the following as a suggestion (not as something already "
            f"decided), writing each value exactly as given:\n{lines}")


def history_messages(history, user, budget=ASSISTANT_HISTORY_TOKENS):
    """The whole conversation as chat messages, newest kept when it outgrows *budget* tokens."""
    from src.memory.tokens import estimate_from_chars
    size = lambda text: estimate_from_chars(len(text or ""))
    msgs = [dict(role="user", content=user)]
    used = size(user)
    for pair in reversed(history):
        cost = size(pair["user"]) + size(pair["assistant"])
        if used + cost > budget:
            break
        msgs[:0] = [dict(role="user", content=pair["user"]), dict(role="assistant", content=pair["assistant"])]
        used += cost
    return msgs


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


def generate_turn(plan, turn, session, history, items, *, user_llm, assistant_llm):
    """One verified pair, or the last attempt with its problems."""
    recent = history[-RECENT_PAIRS:]
    must_say = [dict(about=i["about"], value=i["value"], kind=("paste" if i["paste"] else i["change"]))
                for i in items["user"]]
    problems = []
    for attempt in range(1, MAX_TRIES + 1):
        user = user_llm(USER_SYSTEM, json.dumps(dict(
            PERSONA=plan["persona"], STYLE=STYLES[session["style"]], SESSION_BEATS=session["beats"],
            RECENT=recent, MUST_SAY=must_say,
            NO_MEMORY_TOPIC=(session["topics"][0] if items["quiet"] and not must_say else None)),
            ensure_ascii=False))
        reply = assistant_llm(ASSISTANT_SYSTEM + suggest_note(items["assistant"]), history_messages(history, user))
        problems = ([f"user missing {i['id']}: {i['value']}" for i in items["user"]
                     if i["change"] != "event" and not contains(user, i["value"])]
                    + [f"assistant missing {i['id']}: {i['value']}" for i in items["assistant"]
                       if not contains(reply, i["value"])])
        if not problems:
            return dict(turn=turn, user=user, assistant=reply, attempts=attempt, verified=True)
    return dict(turn=turn, user=user, assistant=reply, attempts=MAX_TRIES, verified=False, problems=problems)


def run(plan, out, *, user_llm, assistant_llm, say=print):
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
    schedule, when = due(plan), session_of(plan)
    history = [dict(user=done[t]["user"], assistant=done[t]["assistant"]) for t in sorted(done)]
    total = len(when)
    for turn in range(1, total + 1):
        if turn in done:
            continue
        session, _ = when[turn]
        items = schedule.get(turn, dict(user=[], assistant=[], quiet=False))
        row = generate_turn(plan, turn, session, history, items, user_llm=user_llm, assistant_llm=assistant_llm)
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


def caller(side):
    kind, _, name = side.partition(":")
    if kind == "local":
        return local_caller(name)
    if kind == "cloud":
        return cloud_caller(name)
    raise ValueError(f"side must be local:<model> or cloud:<role>, got {side!r}")


def cloud_caller(role):
    """(system, prompt) -> text on a Z2 profile (cloud; sized from measured cost first, TRAPS 148)."""
    from experiments.z2.profiles import generator
    gen = generator(role)

    def call(system, prompt):
        turns = prompt if isinstance(prompt, list) else [dict(role="user", content=prompt)]
        result = gen.generate([dict(role="system", content=system)] + turns,
                              temperature=0.8, max_output_tokens=4_000, session_id=f"ice-final-synth-{role}")
        return result.text
    return call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--user", default=USER_SIDE, help="local:<ollama model> | cloud:<role>")
    parser.add_argument("--assistant", default=ASSISTANT_SIDE, help="local:<ollama model> | cloud:<role>")
    args = parser.parse_args()
    plan = json.loads((PLANS / f"{args.plan}.json").read_text())
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "sides.json").write_text(json.dumps(dict(user=args.user, assistant=args.assistant, policy=POLICY)) + "\n")
    status = run(plan, args.out, user_llm=caller(args.user), assistant_llm=caller(args.assistant))
    print(json.dumps(status))
    return 0 if not status["unverified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
