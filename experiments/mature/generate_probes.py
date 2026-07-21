#!/usr/bin/env python3
"""
Generate evaluation probes for each conversation at every split point,
using only the history visible up to that point.

Probe IDs are globally unique within a conversation: {split_turn}-GEN-{N}.

Usage:
  uv run python experiments/mature/generate_probes.py
"""

from typing import List
import json, os, sys, random, uuid, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI

# ---------- Configuration ----------
SEED = 42
random.seed(SEED)

SGLANG_URL = "http://localhost:8003/v1"
SGLANG_MODEL = "mattbucci/gemma-4-12B-AWQ"
MAX_HISTORY_TOKENS = 30_000          # max tokens to send to the model per split
SAMPLING_FIRST_N = 5                 # include first 5 turns in the sample
SAMPLING_LAST_N = 10                 # include last 10 turns in the sample
SAMPLING_RANDOM_N = 15               # include random 15 turns from the middle

SIMULATION_INPUT = Path("data/simulation/simulation_full.jsonl")
MATURE_DIR = Path("experiments/mature")
OUTPUT_FILE = MATURE_DIR / "intermediates" / "generated_probes.json"

TARGET_CONVERSATIONS = [
    {
        "conversation_id": "633e26f8-5889-5c21-8c70-f4d7ab22cb00",
        "label": "Shinchan"
    },
    {
        "conversation_id": "bb558b5f-5365-5bac-9ed0-07219025b5f2",
        "label": "Flaw"
    },
    {
        "conversation_id": "a77c15cf-2078-4279-aeaa-8c3a6d58a972",
        "label": "ICE-Dev"
    },
    {
        "conversation_id": "ecc64aab-1979-5586-b0d8-c53448c0882e",
        "label": "Masters"
    },
]

# ---------- Split logic (copied from run_mature_experiment.py) ----------
def generate_checkpoints(total_turns: int) -> List[int]:
    """Generate reasonably spaced checkpoint turn indices, ending at total_turns."""
    if total_turns <= 0:
        return []

    # Determine number of checkpoints based on conversation length
    if total_turns >= 1000:
        n = random.randint(15, 20)
    elif total_turns >= 500:
        n = random.randint(12, 16)
    elif total_turns >= 200:
        n = random.randint(10, 14)
    else:
        n = random.randint(8, 12)
    n = min(n, total_turns)

    # Evenly spaced base points
    step = total_turns / n
    checkpoints = []
    for i in range(1, n):
        # Random jitter of ±20% around the even step
        jitter = random.uniform(-0.2, 0.2) * step
        cp = int(step * i + jitter)
        cp = max(1, min(cp, total_turns - 1))  # keep in bounds, leave last for final
        checkpoints.append(cp)

    # Remove duplicates and sort
    checkpoints = sorted(set(checkpoints))

    # Always end with total_turns
    if checkpoints and checkpoints[-1] >= total_turns:
        checkpoints[-1] = total_turns
    else:
        checkpoints.append(total_turns)

    # Ensure minimum gap of 5 turns between checkpoints
    cleaned = []
    for cp in checkpoints:
        if not cleaned or cp - cleaned[-1] >= 5:
            cleaned.append(cp)
        else:
            # Merge with previous by skipping
            cleaned[-1] = max(cleaned[-1], cp)
    if cleaned[-1] != total_turns:
        cleaned.append(total_turns)

    return cleaned

def estimate_tokens(text):
    return int(len(text.split()) * 1.33)

# ---------- History sampling ----------
def sample_history(turns, up_to_idx, max_tokens=MAX_HISTORY_TOKENS):
    """Return a representative list of turns (strings) that fits under max_tokens."""
    history_turns = turns[:up_to_idx]
    if len(history_turns) <= (SAMPLING_FIRST_N + SAMPLING_LAST_N + SAMPLING_RANDOM_N):
        # Use all turns
        return [f"User: {t['prompt']}\nAssistant: {t.get('response','')}" for t in history_turns]

    # Always include first N and last N
    sampled = []
    # first N
    for t in history_turns[:SAMPLING_FIRST_N]:
        sampled.append(f"User: {t['prompt']}\nAssistant: {t.get('response','')}")
    # last N
    for t in history_turns[-SAMPLING_LAST_N:]:
        sampled.append(f"User: {t['prompt']}\nAssistant: {t.get('response','')}")

    # random from middle
    middle = history_turns[SAMPLING_FIRST_N:-SAMPLING_LAST_N]
    if middle:
        random_middle = random.sample(middle, min(len(middle), SAMPLING_RANDOM_N))
        for t in random_middle:
            sampled.append(f"User: {t['prompt']}\nAssistant: {t.get('response','')}")

    # Truncate to max_tokens
    total = 0
    result = []
    for s in sampled:
        tok = estimate_tokens(s)
        if total + tok > max_tokens:
            break
        result.append(s)
        total += tok
    return result

# ---------- Probe generation prompt ----------
SYSTEM_PROMPT = """You are an evaluation probe generator for a conversational memory system.
You will receive a conversation history between a user and an AI assistant.
Your task is to generate high‑quality probes that test whether the system correctly remembers
facts, events, character details, decisions, and other information from the history.

The memory system being tested has SEPARATE retrieval mechanisms (a "leg" is one mechanism):
  - CODEX leg: a knowledge graph of named entities and their relationships/properties.
    Best tested by facts that hinge on a specific named entity and a clear relationship or
    attribute (e.g. "who is X's rival", "what is Y's role"). Use names of specific things or 
    person for better promts.
  - VECTOR leg: semantic embedding similarity. Best tested by facts that must be recalled from
    MEANING rather than matching words — the probe should paraphrase the original event using
    different vocabulary than the source text used, with no shared distinctive keywords.
  - BM25 leg: exact lexical/keyword matching. Best tested by facts anchored to a specific,
    distinctive word, name, number, or term that appears verbatim in the source text — a probe
    where finding the right keyword is more useful than understanding the meaning.
  - PROCEDURAL leg: recurring behavioral patterns or preferences shown MULTIPLE times across
    the history, not a single isolated fact (e.g. "what do I usually do when...", "how do I
    normally approach...").

STRICT RULES:

1. PROBES MUST ONLY ASK ABOUT INFORMATION THAT IS EXPLICITLY PRESENT IN THE PROVIDED HISTORY.
   Do NOT ask about future events, later chapters, or anything not yet mentioned.
   If a fact is only partially revealed, phrase the probe to ask about what is known.

2. EVERY PROBE MUST BE TEMPORALLY OR CONTEXTUALLY UNAMBIGUOUS — it must point at ONE specific
   event/fact, not a category of events that could recur multiple times in the story. This is
   the single most important rule. A probe fails this rule if more than one moment in the
   ENTIRE conversation (not just the history seen so far) could plausibly answer it.
   - BAD:  "what were they doing at the end of the day?" — "end of the day" can describe many
     different evenings across a long story; this question has no way to specify WHICH one.
   - GOOD: "what happened right after the math exam when X tried to make Y blush?" — anchors to
     a specific, nameable event that cannot be confused with any other scene.
   - BAD:  "what did the user decide about the database?" — if the user revisits database
     decisions multiple times, this is ambiguous about which decision.
   - GOOD: "what database did the user choose for the session-cache service specifically?" —
     anchors to one named subsystem, not a recurring category of decision.
   - If the only way to ask about an event is with a vague time/category reference, ADD a
     concrete anchor from the source text itself (a named character, item, location, specific
     action, or outcome) so the question cannot be satisfied by a different, later instance of
     the same TYPE of event. If no such anchor exists in the source text, skip that fact and
     choose a different one to probe instead.

3. Generate exactly {COUNT} probes. EACH PROBE MUST TARGET A DIFFERENT RETRIEVAL LEG FROM THE
   LIST ABOVE — no two probes in this batch may target the same leg. Pick whichever {COUNT}
   legs are actually testable given what's in the provided history; skip a leg entirely if the
   history has nothing suitable for it (e.g. don't force a PROCEDURAL probe if no behavior
   repeats). Each probe must include:
   - "probe_id": a short unique label (e.g., "GEN-01", "GEN-02").
   - "target_leg": one of "codex", "vector", "bm25", "procedural" — which mechanism this probe
     is designed to require, per the descriptions above.
   - "target_topic": your best guess at the single topic this probe's source fact belongs to,
     from this fixed list only: Software_&_Tech, STEM_&_Academics, Business_&_Finance,
     Creative_&_Media, Admin_&_Productivity, Lifestyle_&_Health, Social_&_Relationships,
     World_&_Current_Events, Meta_AI, Null_Noise, General_Reference_&_Trivia.
   - "temporal_anchor": a short phrase (5-12 words) naming the SPECIFIC, concrete detail that
     makes this event distinguishable from any other similar event in the story — e.g. "the
     after-exam blush attempt", "the toothpaste prank during the sleepover", "the PostgreSQL
     migration for the session cache". This is a machine-checkable anchor, not narrative
     flourish — if you cannot write a temporal_anchor that uniquely identifies the event, the
     probe is too vague per rule 2 and must be rewritten or dropped.
   - "user_injected_prompt": the question, written as the user would actually type it.
     * Use informal language, occasional typos, abbreviations, missing punctuation.
     * Write as if the user is speaking casually to the AI.
     * The phrasing must be CONSISTENT with target_leg: a "vector" probe paraphrases instead of
       reusing source keywords; a "bm25" probe should naturally include the distinctive keyword
       or name itself; a "codex" probe should name the specific entity directly; a "procedural"
       probe should reference a recurring pattern ("usually", "every time", "again") rather than
       one single instance.
   - "expected_answer": a thorough, accurate answer based ONLY on the provided history.
     * Include all relevant details, names, numbers, decisions.
     * Write in a neutral, factual style. Do NOT add interpretation or speculation.

4. Within the {COUNT} probes, also try to vary surface style across these dimensions where the
   source material allows it — but leg-coverage from rule 3 and anchoring from rule 2 both
   take priority over this:
   - Event recall (e.g., "what happened when Z?")
   - Relationship / social (e.g., "who is friends with whom?") — naturally pairs with a
     "codex" target_leg, since relationships live in the entity graph.
   - Emotional / opinion (e.g., "how did I feel about X?") — naturally pairs with "vector",
     since this system intentionally disables BM25 matching for emotional-intent prompts.
   - Creative / narrative (e.g., "why did the character do that?")
   - Code / technical (e.g., "what library did X depend on?") — naturally pairs with "codex"
     or "bm25", since technical terms are usually exact distinctive keywords.

5. EXAMPLES of good probes (from a different conversation), showing the leg-targeting AND
   temporal anchoring in practice:
   {{
     "probe_id": "EX-01",
     "target_leg": "codex",
     "target_topic": "Social_&_Relationships",
     "temporal_anchor": "shinchan's kendo club rivalry with rika",
     "user_injected_prompt": "who is the rival of shinchan from kendo?",
     "expected_answer": "Shinchan's rival in the Kendo club is Rika Miyamoto, who serves as the vice-captain."
   }},
   {{
     "probe_id": "EX-02",
     "target_leg": "vector",
     "target_topic": "Creative_&_Media",
     "temporal_anchor": "kael's renaming to aroh",
     "user_injected_prompt": "wasnt kael called smth else before? what changed",
     "expected_answer": "The character originally named Kael was later renamed to Aroh. The name change occurred in a later update to the story."
   }},
   {{
     "probe_id": "EX-03",
     "target_leg": "bm25",
     "target_topic": "Creative_&_Media",
     "temporal_anchor": "the fake wedding staged for ai's grandmother",
     "user_injected_prompt": "so do you rememeber the fake wedding why did that happen??",
     "expected_answer": "The fake wedding was orchestrated because Ai's grandmother wished to see her granddaughter married before she died. The group staged the event to fulfill her wish."
   }},
   {{
     "probe_id": "EX-04",
     "target_leg": "procedural",
     "target_topic": "Software_&_Tech",
     "temporal_anchor": "recurring debugging-first-reproduce pattern, not one instance",
     "user_injected_prompt": "what do i usually do first whenever i start debugging a new module",
     "expected_answer": "The user's recurring debugging pattern is to first reproduce the bug in isolation with a minimal script before touching the main codebase."
   }}

6. If the provided history genuinely cannot support {COUNT} distinct-leg probes (e.g. there is
   no recurring behavior for "procedural", or no distinctive keyword for "bm25"), generate fewer
   probes rather than forcing a bad fit — just return fewer objects, with no explanation.

7. OUTPUT FORMAT: Return ONLY a valid JSON array of probe objects. No other text, no markdown,
   no explanation.

Now, generate probes for the following conversation history:"""

ALL_LEGS = ["codex", "vector", "bm25", "procedural"]
MIN_LEGS_PER_SPLIT = 2
MAX_LEGS_PER_SPLIT = 4

# ---------- Main ----------
def main():
    os.makedirs(MATURE_DIR / "intermediates", exist_ok=True)
    client = OpenAI(base_url=SGLANG_URL, api_key="dummy")

    # Load simulation data
    with open(SIMULATION_INPUT, "r") as f:
        all_turns = [json.loads(line) for line in f if line.strip()]
    conv_turns = defaultdict(list)
    for t in all_turns:
        conv_turns[t.get("conversation_id")].append(t)

    # Load existing generated probes if any (for resume)
    all_probes = {}
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                all_probes = json.load(f)
        except Exception:
            all_probes = {}

    for conv_cfg in TARGET_CONVERSATIONS:
        cid = conv_cfg["conversation_id"]
        label = conv_cfg["label"]
        turns = conv_turns.get(cid, [])
        if not turns:
            print(f"⚠️  No turns for {label}")
            continue

        turns.sort(key=lambda x: x.get("timestamp", ""))
        checkpoints = generate_checkpoints(len(turns))
        print(f"📊 {label}: {len(turns)} turns, {len(checkpoints)} checkpoints: {checkpoints}")

        # Overwrite this conversation's probes entirely (fresh start)
        all_probes[cid] = {}

        for split_turn in checkpoints:
            print(f"  Generating probes for split at turn {split_turn}...")
            history_sample = sample_history(turns, split_turn)
            if not history_sample:
                all_probes[cid][str(split_turn)] = []
                continue

            history_text = "\n\n".join(history_sample)
            probe_count = random.randint(MIN_LEGS_PER_SPLIT, MAX_LEGS_PER_SPLIT)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(COUNT=probe_count)},
                {"role": "user", "content": f"Conversation history:\n\n{history_text}"}
            ]

            # ── Retry loop ──
            success = False
            probes = []
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=SGLANG_MODEL,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=3000,
                        timeout=120.0,
                    )
                    raw = resp.choices[0].message.content.strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                        raw = raw.strip()
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                    probes = parsed
                except Exception as e:
                    print(f"    Attempt {attempt+1} failed: {e}")
                    time.sleep(2)
                    continue

                # Validate and assign clean sequential IDs
                validated = []
                seen_legs = set()
                for p in probes:
                    leg = p.get("target_leg")
                    if leg not in ALL_LEGS:
                        print(f"    ⚠️  Dropping probe: invalid target_leg={leg!r}")
                        continue
                    anchor = (p.get("temporal_anchor") or "").strip()
                    if len(anchor.split()) < 3:
                        print(f"    ⚠️  Dropping probe: temporal_anchor too short ({anchor!r})")
                        continue
                    if leg in seen_legs:
                        print(f"    ⚠️  Duplicate target_leg={leg!r} (keeping anyway)")
                    seen_legs.add(leg)
                    p["split_turn"] = split_turn
                    p["conversation_id"] = cid
                    validated.append(p)

                if len(validated) >= MIN_LEGS_PER_SPLIT:
                    # ── Assign clean sequential IDs per split ──
                    for idx, p in enumerate(validated, start=1):
                        p["probe_id"] = f"{split_turn}-GEN-{idx:02d}"
                    probes = validated
                    success = True
                    break
                else:
                    print(f"    Only {len(validated)} valid probes after validation, retrying...")

            if not success:
                print(f"    ❌ Failed to generate valid probes after 3 attempts, storing empty list.")
                probes = []

            all_probes[cid][str(split_turn)] = probes
            print(f"    ✅ Saved {len(probes)} probes")

            # Save incrementally
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_probes, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All probes saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()