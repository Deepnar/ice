#!/usr/bin/env python3
"""Call Gemma-4-12B-AWQ to extract named entities from each turn.
Supports graceful shutdown and cancellation of pending tasks."""

import asyncio
import json
import os
import signal
import sys
from aiohttp import ClientSession, ClientTimeout, ClientError
from tqdm.asyncio import tqdm

INPUT = "data/ner/raw_turns.jsonl"
OUTPUT = "data/ner/extracted_entities.jsonl"
SGLANG_URL = "http://localhost:8003/v1/chat/completions"
MODEL_NAME = "mattbucci/gemma-4-12B-AWQ"
CONCURRENCY = 8
TIMEOUT = 60

SYSTEM_PROMPT = """You are an entity extraction system. You extract named entities from a single conversation turn (narrative/story content, code/technical content, or both) so they can be located in the original text via case-insensitive substring search and converted into BIO training labels.

═══════════════════════════════════════════
RULE 0 (HIGHEST PRIORITY): EXTRACT VERBATIM
═══════════════════════════════════════════
Every string you output MUST appear in the source text as an exact substring (case differences are OK, nothing else is).


Do NOT fix spelling. "Keal" stays "Keal". "Fastapi" stays "Fastapi" even if the real name is "FastAPI".
Do NOT normalize whitespace, punctuation, or pluralization. If the text says "goo  blade" (two spaces), output "goo  blade", not "goo blade".
Do NOT expand, abbreviate, or paraphrase. If the text says "the Order", output "the Order" or "Order" only if that exact casing/spelling appears — never invent a cleaner form.
Do NOT translate or convert code style. "get_data" stays "get_data", not "getData", even if you think that's "more correct".
If you are not 100% sure the exact characters you're about to output appear in the source, do not output them.


═══════════════════════════════════════════
WHAT TO EXTRACT
═══════════════════════════════════════════
Use these categories as a thinking guide for what counts as "named" — do NOT include the category label in your output, only the entity string itself.

NARRATIVE:


PERSON: character names, named beings, named entities-as-characters ("Kael", "Lethe", "Aroh", "the Boy" if used as a proper epithet/name)
LOCATION: named places ("Binary Universe", "Earth", "Caelestia")
ORGANIZATION: named groups/factions ("The Order", "Tech Guild")
EVENT: named events ("The Great War", "The Corruption")
OBJECT/CONCEPT: named items or named abstract concepts ("goo blade", "Infinity Engine", "The Codex", "Proof of Work" if used as a named strategy)


CODE/TECHNICAL:


FUNCTION/METHOD: named callables, as written in the text, including parens if the text includes them ("render()", "calculate_score", "fetch_data", "async def get_data():")
CLASS: named classes/types ("FastAPI", "SQLAlchemy", "DataLoader")
LIBRARY/MODULE/FRAMEWORK/PROJECT: named packages or projects ("torch", "numpy", "celery", "ICE", "Vue.js")
VARIABLE/CONSTANT: named identifiers that are specific, not generic ("API_KEY", "DEBUG_MODE")
FILE/PATH (when clearly a specific named file): ("config.yaml", "main.py")


Multi-word and nested entities are both fine — extract at whatever granularity appears in the text. If "Binary Universe" and "Universe" both appear as meaningful standalone mentions, you may extract both; don't force a single choice.

═══════════════════════════════════════════
PLACEHOLDER REFERENCES (no name yet, still an entity)
═══════════════════════════════════════════
Many works-in-progress use descriptive labels — "the 2nd protagonist", "villainess", "the serpent", "goo" — because a name hasn't been chosen yet. These ARE entities if they function as a stable reference to ONE specific character/place/object across the conversation, even though the words themselves are generic and lowercase.

The test: if you swapped in a real name later, would the sentence mean the same thing about the same specific character? If yes → extract it. If the word is making a general statement true of any member of that category → do not extract it.


"the 2nd protagonist is afraid of the serpent" (referring to specific characters in the user's story) → extract "2nd protagonist" and "serpent"
"a protagonist needs a flaw to be interesting" (general craft statement, not pointing at anyone specific) → do NOT extract "protagonist"
"the villainess finally meets goo" (specific character + specific named-ish object in the user's world) → extract "villainess" and "goo"
"every villainess needs a motive" (general statement) → do NOT extract "villainess"


When a placeholder term repeats across the conversation referring to the same character, that repetition is itself a signal it's an entity, not a generic mention — extract it consistently each time it's used that way.

ABSTRACT-CONCEPT PLACEHOLDERS — extra care needed:
Stories sometimes personify abstract forces ("an entity of death", "the life entity", "the balance"). The same swap-test applies, but be stricter here, because words like "life", "death", "existence", "universe", "balance" overwhelmingly appear in ordinary, non-referring use even within the same story-planning conversation.


Extract the full phrase when it's clearly naming a specific personified force or entity: "the life entity", "an entity of death", "the death entity" → extract "life entity" / "entity of death" / "death entity" (whichever exact wording appears).
Do NOT extract the bare abstract noun on its own when it's just describing what something does, is about, or affects: "creates life and universes" → do not extract "life" or "universes"; "balancing life and death" → do not extract "life" or "death"; "destroying existence" → do not extract "existence"; "trying to find the correct balance in the universe" → do not extract "universe".
The test: would removing the word leave an entity-shaped hole ("the ___ entity"), or would it just leave a sentence with one less adjective/object ("creates life and ___")? Only the first case is an entity.
If you're extracting "life" or "death" or "universe" or "existence" as a bare single word with no entity-marking word attached (like "entity"), stop — this is very likely a false positive. These four words in particular should almost never be extracted alone.


═══════════════════════════════════════════
SPAN BOUNDARIES — do not include leading/trailing function words
═══════════════════════════════════════════
An extracted span must be a coherent noun phrase, never a sentence fragment. Do NOT include ordinals, numbers, or connecting words unless they are integral to how the placeholder/name is actually used as a label.


"1st set up a protagonist..." → do NOT extract "1st set up" (this is a verb phrase fragment, not an entity, even though it starts with a number)
"the 2nd protagonist" → DO extract "2nd protagonist" (here the ordinal is integral to how the user labels this specific character, distinguishing it from "1st protagonist")
The difference: in the second case, removing the ordinal would make the reference ambiguous between two characters the user is tracking. In the first case, the ordinal is just describing sequence in a sentence ("first, [verb phrase]..."), not labeling a character.
When in doubt, extract the noun phrase only, drop any leading verb or connector word stuck to it by the sentence's grammar.


═══════════════════════════════════════════
WHAT NOT TO EXTRACT
═══════════════════════════════════════════


Generic/common nouns used in their ordinary, non-referring sense: "a friend", "the enemy" (as a category), "this world" (as in "the real world"), "the universe" (as in "the cosmos" generally), "story", "code", "function", "error", "bug" — unless they're acting as the placeholder-reference described above, or are capitalized/used as an actual proper name
Descriptive phrases with no referent: "trial and tribulation", "cosmic balance", "life and death"
Emotion/state words: "sad", "frustrated", "happy"
Placeholder roles for the chat participants themselves: "User", "Assistant", "the user", "the AI" (this is different from in-story character placeholders, which ARE extracted per the section above)
Pronouns, even when clearly referring to a named entity: "he", "she", "it", "they"
Single generic English words even if capitalized only by sentence position (e.g., a sentence starting "Trial and error..." — "Trial" is not an entity here)


When genuinely uncertain whether something is a specific referring expression or just a description, do NOT extract it. Precision over recall — a missed entity is recoverable, a wrong one corrupts training data. But do not default to excluding placeholder character/place references just because they're lowercase or look like common words — check whether they're doing referential work first.

═══════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════
Output ONLY a JSON array of strings. No markdown, no code fences, no explanation, no reasoning text before or after.
If no named entities are found, output [].
Duplicate mentions of the same entity in one turn: include the string once (the alignment step finds all occurrences itself).

═══════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════

Example 1 — Narrative, clean:
User: "Kael and Lethe are fighting in the Binary Universe."
Assistant: "Yes, Kael is using the goo blade."
Output: ["Kael", "Lethe", "Binary Universe", "goo blade"]

Example 2 — Narrative, misspelling preserved verbatim:
User: "Keal raised the goo blade against the Order, thinking of the Corruption."
Output: ["Keal", "goo blade", "Order", "Corruption"]
(Note: "Keal" is kept misspelled — never corrected to "Kael".)

Example 3 — Narrative, lowercase/non-capitalized named terms:
User: "the witherlight isn't just a weapon, it's what binds aroh to the eden saga."
Output: ["witherlight", "aroh", "eden saga"]
(Note: lowercase proper nouns are still extracted — capitalization is not the test, specificity is.)

Example 4 — Code, Python:
User: "I'm using FastAPI with SQLAlchemy, and I keep getting this error."
Assistant: "Try using 'async def get_data():' instead of a regular function."
Output: ["FastAPI", "SQLAlchemy", "get_data()"]

Example 5 — Code, misspelled library name preserved:
User: "I imported Fastapi wrong, it should be 'from fastapi import FastAPI' not 'from Fastapi import App'."
Output: ["Fastapi", "fastapi", "FastAPI", "App"]
(Note: "Fastapi" and "fastapi" both appear as distinct substrings in the source and are both kept, unmerged, uncorrected.)

Example 6 — Code, JS/TS style:
User: "My useEffect hook in Dashboard.tsx keeps calling fetchUserData twice, and I think it's because of the QueryClient setup in react-query."
Output: ["useEffect", "Dashboard.tsx", "fetchUserData", "QueryClient", "react-query"]

Example 7 — Mixed, project + tools:
User: "In my project ICE, I'm using Celery with Redis for background tasks."
Assistant: "That's a solid setup for distributed queues."
Output: ["ICE", "Celery", "Redis"]

Example 8 — Nested/overlapping entities:
User: "The Infinite Context Engine, or ICE, finally passed the eval — the Vector RAG part of ICE was the bottleneck."
Output: ["Infinite Context Engine", "ICE", "Vector RAG"]

Example 9 — No entities:
User: "this is sad"
Assistant: "I understand how frustrating it can be..."
Output: []

Example 10 — No entities despite capitalization:
User: "Honestly I'm Done with this, nothing is Working today."
Output: []
(Note: "Done" and "Working" are capitalized only by emphasis/sentence position, not named entities.)

Example 11 — Placeholder character references (no name chosen yet), specific:
User: "the 2nd protagonist still hasn't forgiven the villainess, even after the serpent showed up and goo started spreading through the city. The story flaw cannot have this."
Output: ["2nd protagonist", "villainess", "serpent", "goo","flaw"]
(Note: none of these are capitalized or "proper nouns" in the dictionary sense, but each refers to one specific character/object in the user's story — they are extracted as entities.)

Example 12 — Same words, generic/non-referring use, NOT extracted:
User: "I think every story needs a strong protagonist and a villainess with a real motive, otherwise the conflict feels hollow. My story FLAW needs this"
Output: ["FLAW"]
(Note: "protagonist" and "villainess" here are general craft statements about story-writing, not references to any specific character — contrast with Example 11.)

Example 13 — Mixed: one placeholder entity, one generic mention, same turn:
User: "My villainess (still unnamed) just discovered the serpent isn't actually an enemy — funny how every story needs an enemy like that to work."
Output: ["villainess", "serpent"]
(Note: the first "enemy" describing the serpent's specific role is borderline but skip it since "isn't actually an enemy" is a negation about category membership, not a referring expression; the second "enemy" in "every story needs an enemy" is explicitly generic and excluded. "villainess" and "serpent" are kept because they consistently refer to specific story elements.)

Example 14 — Abstract-concept placeholders, entity form vs bare noun, same sentence:
User: "his connection to an entity which creates life and universes. his friend gets connected to a counterpart of the life entity, an entity of death."
Output: ["life entity", "entity of death"]
(Note: "life" and "universes" in "creates life and universes" are bare abstract nouns describing what the entity does — NOT extracted. "life entity" and "entity of death" ARE extracted because "entity" turns them into a labeled, specific referent. Never extract "life", "death", "universe", or "existence" alone — they should almost always be excluded as standalone words.)

Example 15 — Span boundaries, ordinal as fragment vs ordinal as label:
User: "1st set up a protagonist lost in time... the 2nd protagonist wants to find the solution, but nobody can determine this about the 1st protag."
Output: ["2nd protagonist", "1st protag"]
(Note: "1st set up" is NOT extracted — it's a verb-phrase fragment ("first, [the story] set up..."), not a character label. "2nd protagonist" and "1st protag" ARE extracted — here the ordinal distinguishes between two specific characters the user is tracking, so it's integral to the reference.)

Example 16 — Generic abstract noun even mid story-planning:
User: "he wants to protect existence, but every time the universe resets, the same outcome happens."
Output: []
(Note: "existence" and "universe" are used here in their ordinary sense — there's no entity-marking word attached, and nothing distinguishes "the universe" from "any universe in general". Contrast with Example 14, where "entity" turns the same kind of word into a specific referent.)
"""

shutdown_requested = False
current_tasks = set()  # Track all running tasks
write_lock = asyncio.Lock()

def signal_handler(sig, frame):
    global shutdown_requested
    if not shutdown_requested:
        shutdown_requested = True
        print("\n⚠️ Shutdown requested. Cancelling pending tasks...")
        # Cancel all tracked tasks
        for task in current_tasks:
            if not task.done():
                task.cancel()
    else:
        print("\n⚠️ Force exit...")
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def extract_entities(session, sem, item):
    """Extract entities from a single turn. Cancellable."""
    global shutdown_requested
    
    if shutdown_requested:
        raise asyncio.CancelledError()
    
    user_msg = f"Extract all named entities from the following conversation turn:\n\n{item['raw_text']}"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.0,
        "stream": False
    }
    
    async with sem:
        for attempt in range(3):
            if shutdown_requested:
                raise asyncio.CancelledError()
            try:
                # Use wait_for to enforce timeout and allow cancellation
                async with session.post(SGLANG_URL, json=payload, timeout=ClientTimeout(total=TIMEOUT)) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"SGLang error {resp.status}: {text}")
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                        content = content.strip()
                    return json.loads(content)
            except asyncio.CancelledError:
                raise  # Propagate cancellation
            except (asyncio.TimeoutError, ClientError) as e:
                if attempt == 2:
                    print(f"⚠️ Failed after 3 attempts for {item.get('id', 'unknown')}: {e}")
                    return []
                await asyncio.sleep(2 * (attempt + 1))
            except Exception as e:
                if attempt == 2:
                    print(f"⚠️ Unexpected error for {item.get('id', 'unknown')}: {e}")
                    return []
                await asyncio.sleep(2 * (attempt + 1))
        return []

async def process_one(session, sem, item, pbar):
    """Process a single item, save results, update progress."""
    global shutdown_requested
    if shutdown_requested:
        raise asyncio.CancelledError()
    
    entities = await extract_entities(session, sem, item)
    if entities is not None:
        item["entities"] = entities
        async with write_lock:
            with open(OUTPUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
    pbar.update(1)

async def main():
    global shutdown_requested, current_tasks
    
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    
    # Read input
    with open(INPUT, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    print(f"📄 Loaded {len(all_items)} records from {INPUT}")
    
    # Resume from existing output
    processed_ids = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        processed_ids.add(obj.get("id"))
                    except:
                        pass
        print(f"🔄 Resuming: {len(processed_ids)} already processed")
    
    items = [item for item in all_items if item.get("id") not in processed_ids]
    total = len(items)
    print(f"📊 {total} records remaining to process")
    
    if total == 0:
        print("✅ All records already processed!")
        return
    
    sem = asyncio.Semaphore(CONCURRENCY)
    
    # Progress bar
    with tqdm(total=total, desc="Extracting entities") as pbar:
        async with ClientSession() as session:
            # Create tasks and track them
            tasks = []
            for item in items:
                if shutdown_requested:
                    break
                task = asyncio.create_task(process_one(session, sem, item, pbar))
                tasks.append(task)
                current_tasks.add(task)
                task.add_done_callback(current_tasks.discard)
            
            if not tasks:
                return
            
            # Wait for tasks with cancellation support
            try:
                await asyncio.gather(*tasks, return_exceptions=False)
            except asyncio.CancelledError:
                # This will be raised when we cancel tasks
                print("\n⏹️ Cancelled remaining tasks. Waiting for cleanup...")
                # Wait a moment for tasks to finish cancellation
                await asyncio.sleep(1)
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")
            
            # Final save: ensure all completed results are written (already done incrementally)

    print(f"\n✅ Entity extraction complete → {OUTPUT}")
    if shutdown_requested:
        print("   ⚠️ Processing was interrupted. Resume by re-running.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        sys.exit(0)
    except asyncio.CancelledError:
        print("\n⚠️ Tasks cancelled.")
        sys.exit(0)