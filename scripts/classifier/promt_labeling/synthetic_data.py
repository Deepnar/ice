import asyncio
import csv
import json
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import json.decoder
import re  # add at the top of the file
# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
BASE_URL = "http://localhost:8001/v1"
API_KEY = "dummy"

CONCURRENT_ROWS = 5              # process 5 CSV rows at once
BATCH_SIZE = 5                   # generate this many prompts per API call

INPUT_PATH = "/home/deepnar/Programs/ice/scripts/classifier/promt_labeling/synth_promt_gen_number.csv"
OUTPUT_PATH = "/home/deepnar/Programs/ice/data/synthetic/synthetic_prompts_renumbered_labeled.jsonl"
FAILED_PATH = "/home/deepnar/Programs/ice/data/synthetic/failed_synthetic_prompts.jsonl"

# =====================================================================
# RAW ASYNC CLIENT (no instructor – we want plain text)
# =====================================================================
raw_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# =====================================================================
# TRIMMED GENERATION SYSTEM PROMPT
# =====================================================================
GENERATION_SYSTEM_PROMPT = """You are a synthetic training data generator for an AI memory classifier. You must produce realistic, human-written user prompts that a real person would send to a chatbot.

You will receive: Source, Topic, Intent, Context_Reliance, and a count.
Return EXACTLY that many prompts as a JSON array of strings. Nothing else. No markdown, no explanation.

═══════════════════════════════════════════════════════
VOICE & TONE (all prompts)
═══════════════════════════════════════════════════════
- Use casual, conversational language. Write like a real person texting or asking a quick question.
- Include occasional typos, missing punctuation, lower-case, contractions (dont, im, ur, idk, tbh).
- Do NOT start with “Can you help me with…” or “Please”.
- Every prompt must feel specific and genuine, never robotic or benchmark-like.

═══════════════════════════════════════════════════════
CONTEXT RELIANCE – the most important rule
═══════════════════════════════════════════════════════

── Long_Term_Memory (LTM) ──
The prompt MUST be answerable ONLY if the AI remembers past conversations. The user assumes shared context. Every LTM prompt must contain at least one of these signals:
- “we talked about”, “we decided”, “the plan we made”
- **Personal names / possessives**: “my friend [Name]”, “my character [Name]”, “my colleague [Name]” (Signal D)
- “the [project/bug/story/plan] we [built/fixed/designed]”
- “u said”, “you mentioned”, “you suggested”
- “that [thing/bug] from [yesterday/last week/last time]”
- “remember when / remember that”
- “continue from where we left off”
- Implicit anaphora: “make it shorter”, “that thing you said”, “will it work this time” – a reference that is meaningless without memory of previous exchanges
- Named entities with no introduction – the AI is expected to already know “the goo blade”, “the ICE system”, “my friend Priya”, “the newsletter launch”

BAD LTM: “Can you help me fix a bug?” ← no memory signal
GOOD LTM: “that segfault we kept hitting last night is back, i thought we fixed it when we changed the malloc call”

── Real_Time_Search (RTS) ──
The prompt MUST require live, current, or time-sensitive information that a static model cannot answer. Every RTS prompt must explicitly use one of these temporal cues:
- “today”, “tomorrow”, “this week”, “right now”, “latest”, “current”
- “just happened”, “this morning”, “is X out yet”, “did X drop”
- “who won the match today”, “what’s the score right now”
- “weather forecast for tomorrow”, “temperature right now”

BAD RTS: “What is the latest Python version?” ← sounds like trivia
GOOD RTS: “is python 3.13 out yet or still rc? about to set up a new venv and dont wanna use something broken”

── Zero_Shot ──
Self-contained; no memory or live data needed. Must NOT sound like a trivia question. Reframe “What is X?” as genuine curiosity.
BAD: “What is the capital of France?”
GOOD: “wait is it true that a day on venus is longer than its year? i heard that somewhere and it sounds fake”

═══════════════════════════════════════════════════════
TOPIC & INTENT GUIDE – only the labels you will generate
═══════════════════════════════════════════════════════

── Software_&_Tech ──
Real tech names, error messages, config details, coding tasks.
GOOD (Generation): “write me a python decorator that retries a function up to 3 times with exponential backoff, i keep writing this from scratch and its annoying”
GOOD (Troubleshooting): “my docker container keeps OOMing after about 20 mins even tho the service barely uses memory when i run it locally”
GOOD (Factual_Retrieval): “why does pip install torch always pick the cpu version even when i have cuda installed”

── Creative_&_Media ──
Stories, characters, worldbuilding, writing scenes. Emotional attachment to the work.
GOOD (Generation): “i want the villain’s monologue to feel like something out of a greek tragedy but without sounding pretentious”
GOOD (Generation with LTM): “that scene where the villainess gives him the goo blade, can we make it so she hands it over without looking at him, like we said last time”

── Lifestyle_&_Health ──
Personal habits, fitness, mental health, daily routines. Specific to their situation.
GOOD (Emotional_Processing): “i bombed the interview and i keep replaying every stupid answer i gave, i hate that i freeze under pressure”
GOOD (Emotional_Processing with LTM): “ive been sleeping 4 hours a night for like 2 weeks and my brain feels like static, this is exactly what happened during exams last year”

── Business_&_Finance ──
Startups, money, career, pricing, decision-making. Specific numbers or timelines.
GOOD (Strategic_Planning): “i want to go from zero to deployed in 2 weeks for this mvp, what should i tackle first given i have about 3 hours a day”
GOOD (Strategic_Planning with LTM): “the launch plan we mapped out last month had us doing the cold email campaign first – should we still start there or pivot to the influencer thing u suggested?”

── General_Reference_&_Trivia ──
Casual curiosity about random facts, definitions. Never a quiz question.
GOOD (Factual_Retrieval): “what even is a quasar, i saw it mentioned in a meme and now im curious”
GOOD (Factual_Retrieval with LTM): “what was that russian composer we talked about whose works all sound like winter? i keep forgetting his name”
GOOD (Casual_Banter): “haha that joke was terrible lmaoo”

── World_&_Current_Events ──
News, politics, regional events, public figures. Includes live-data requests.
GOOD (Factual_Retrieval RTS): “did the supreme court issue any rulings this week on privacy?”
GOOD (Factual_Retrieval RTS): “what’s the latest on that flooding in bangkok? is the airport open”

── STEM_&_Academics ──
Math, science, research, studying. May include formulas or paper references.
GOOD (Factual_Retrieval): “my prof said that p=np is basically unsolvable but then why are there still new papers claiming to solve it every year”
GOOD (Factual_Retrieval RTS): “did the arxiv paper on rope scaling get updated today?”

── Null_Noise ──
Gibberish, accidental sends, test messages. ONLY short, meaningless strings under 20 characters. Vary the gibberish – random letters, numbers, punctuation, nonsense mash. Never repeat the same exact string.
GOOD: “asdfghjkl”, “zzzzzz”, “12345”, “....”, “test”, “ghghgh”, “ahhhh”, “lmao”, “hmmm”, “noooo”, “yesss”, “uhhh”, “okokok”, “blabla”
BAD: any sentence or coherent thought.

── Casual_Banter ──
Social greetings, thanks, jokes, playful messages. No task or question.
GOOD: “yo good morning, hows it going”
GOOD: “thanks that actually helped a ton appreciate it”
GOOD: “haha that was such a dumb joke lmaooo”

── Troubleshooting ──
Something is broken; user wants it fixed. Error messages, unexpected behavior.
GOOD: “my docker container keeps OOMing after about 20 mins even tho the service barely uses memory when i run it locally”
GOOD (with LTM): “that bug we fixed last night is back, the authentication module still crashes”

── Generation ──
User wants something created or produced: code, story, document, names.
GOOD: “write me a python decorator that retries a function up to 3 times”
GOOD (with LTM): “continue from where we left off, i want the next scene to start right after the hug”

── Emotional_Processing ──
Venting, feelings, seeking validation. No information request.
GOOD: “i bombed the interview and i keep replaying every stupid answer i gave”
GOOD (with LTM): “i tried the morning routine u suggested and i still cant get out of bed, i feel like its never gonna change”

── Strategic_Planning ──
Wants a plan, roadmap, or approach.
GOOD: “help me plan a 30-day learning path to learn machine learning”
GOOD (with LTM): “we said we’d build the retrieval pipeline next, but now im thinking we should do the memory slots first – whats your take?”

── Factual_Retrieval ──
Wants a specific fact or explanation. Must be curious, not a quiz question.
GOOD: “wait is it true that you lose most heat through your head or is that debunked”
GOOD (with LTM): “what was the architecture decision we made for the retrieval system? i need to write it up”

═══════════════════════════════════════════════════════
ABSOLUTE FORBIDDEN PATTERNS
═══════════════════════════════════════════════════════
- No “Can you help me with…” or “Please” + formal request.
- No trivia‑style “What is X?” without casualization.
- No benchmark/textbook sounding language.
- For Null_Noise: only short gibberish, no sentences.
- For RTS: you MUST use a temporal word like “today”, “this week”, “right now”.
- For LTM: you MUST include a clear memory signal (see the list above).

═══════════════════════════════════════════════════════
SELF-CHECK BEFORE OUTPUT
═══════════════════════════════════════════════════════
1. Does the prompt sound like a real person?
2. If LTM: is there at least one memory signal?
3. If RTS: is there a temporal cue?
4. If Null_Noise: is it under 20 chars and meaningless?
5. Would a real person plausibly type this into a chatbot?

OUTPUT FORMAT: a single valid JSON array of strings. No markdown, no preamble. Start with [ and end with ].
"""

# =====================================================================
# ASYNC WORKER FOR ONE ROW (with internal batching)
# =====================================================================
# =====================================================================
# PLACEHOLDER REASONING FOR SYNTHETIC DATA
# =====================================================================
REASONING_PLACEHOLDER = "Synthetic generation with expected labels."


async def generate_row(
    semaphore: asyncio.Semaphore,
    row: dict,
    outfile,
    failed_file,
    pbar: tqdm,
    id_counter: list   # mutable list with one integer
):
    async with semaphore:
        source = row["Source"]
        topic = row["Topic"]
        intent = row["Intent"]
        count = int(row["Count"])
        context = row["Context_Reliance"]

        topic_labels = [topic]
        intent_labels = [intent]

        generated = 0
        while generated < count:
            remaining = count - generated
            batch_size = min(BATCH_SIZE, remaining)

            user_message = (
                f"Source: {source}\n"
                f"Topic: {topic}\n"
                f"Intent: {intent}\n"
                f"Context reliance: {context}\n\n"
                f"Generate exactly {batch_size} realistic user prompts that fit these labels. "
                "Return ONLY a valid JSON array of strings, like [\"prompt1\", \"prompt2\", ...]. "
                "Do NOT include any introductory or concluding text."
            )

            try:
                response = await raw_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                )

                raw_text = response.choices[0].message.content.strip()
                if raw_text.startswith("```"):
                    parts = raw_text.split("```")
                    raw_text = parts[1] if len(parts) > 1 else parts[0]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                    raw_text = raw_text.strip()

                # Parse the first complete JSON array, ignoring anything after
                decoder = json.JSONDecoder()
                try:
                    prompts, end_idx = decoder.raw_decode(raw_text)
                except json.JSONDecodeError:
                    start = raw_text.find('[')
                    if start != -1:
                        prompts, end_idx = decoder.raw_decode(raw_text, start)
                    else:
                        raise ValueError("No JSON array found in response")

                if not isinstance(prompts, list):
                    raise ValueError("Model did not return a JSON array")

                # Write with unique IDs
                for prompt in prompts:
                    labeled_entry = {
                        "id": f"synth_{id_counter[0]}",
                        "source": source,
                        "prompt": prompt,
                        "label": {
                            "reasoning": REASONING_PLACEHOLDER,
                            "topic_labels": topic_labels,
                            "intent_labels": intent_labels,
                            "context_reliance": context
                        }
                    }
                    outfile.write(json.dumps(labeled_entry) + "\n")
                    outfile.flush()
                    id_counter[0] += 1
                    generated += 1
                    pbar.update(1)

            except Exception as e:
                tqdm.write(f"❌ FAILED batch {source}-{topic}-{intent}: {e}")
                failed_file.write(json.dumps({
                    "source": source,
                    "topic": topic,
                    "intent": intent,
                    "context": context,
                    "batch_size": batch_size,
                    "error": str(e)
                }) + "\n")
                failed_file.flush()
                pbar.update(batch_size)
                break

# ----------------------------------------------------------------------
# Helper: count existing prompts per combo and find max synth ID
# ----------------------------------------------------------------------
def analyze_existing_output(output_path):
    """Return (combo_counts, next_id) where combo_counts is a dict
    mapping (source, topic, intent, context) -> count, and next_id is
    the next available synth_ ID (max existing + 1)."""
    combo_counts = {}
    max_id = -1

    if not os.path.exists(output_path):
        return combo_counts, 0

    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Extract key fields
                src = entry.get("source", "")
                lbl = entry.get("label", {})
                tp = lbl.get("topic_labels", [])
                it = lbl.get("intent_labels", [])
                ctx = lbl.get("context_reliance", "")

                # Create a combo key (topic and intent are lists, take first element)
                tp_key = tp[0] if tp else ""
                it_key = it[0] if it else ""
                combo_key = (src, tp_key, it_key, ctx)
                combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1

                # Track max ID
                eid = entry.get("id", "")
                if eid.startswith("synth_"):
                    try:
                        num = int(eid.split("_")[1])
                        if num > max_id:
                            max_id = num
                    except (IndexError, ValueError):
                        pass
            except (json.JSONDecodeError, KeyError):
                continue

    next_id = max_id + 1
    return combo_counts, next_id


# ----------------------------------------------------------------------
# Updated main()
# ----------------------------------------------------------------------
async def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Cannot find input file at '{INPUT_PATH}'")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FAILED_PATH), exist_ok=True)

    # Analyze existing output
    combo_counts, next_id = analyze_existing_output(OUTPUT_PATH)
    print(f"📊 Existing output: {sum(combo_counts.values())} prompts, next ID will be {next_id}")

    # Read CSV rows
    rows = []
    total_prompts = 0
    with open(INPUT_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            total_prompts += int(row["Count"])

    # Filter and adjust counts
    adjusted_rows = []
    skipped = 0
    for row in rows:
        src = row["Source"]
        tp = row["Topic"]
        it = row["Intent"]
        ctx = row["Context_Reliance"]
        needed = int(row["Count"])
        existing = combo_counts.get((src, tp, it, ctx), 0)
        remaining = needed - existing
        if remaining <= 0:
            skipped += needed
            continue
        # Create a new dict with the adjusted count
        adj_row = dict(row)
        adj_row["Count"] = remaining
        adjusted_rows.append(adj_row)
        total_prompts = total_prompts - existing  # update total to generate

    print(f"📋 {len(adjusted_rows)} rows to process, {total_prompts} prompts to generate (skipped {skipped} already done)")

    if total_prompts == 0:
        print("✅ All prompts already generated. Nothing to do.")
        return

    # Create a custom progress bar that starts at next_id
    semaphore = asyncio.Semaphore(CONCURRENT_ROWS)

    # We need to pass next_id into the workers – we'll use a mutable object like a list
    id_counter = [next_id]

    # We'll adapt generate_row to accept id_counter (modify signature slightly)
    # So we need to update the function definition to take id_counter as argument.
    # Because we can't modify generate_row in-place without showing whole code, I'll
    # provide the new signature and body.

    # (See updated generate_row below)

    with open(OUTPUT_PATH, "a") as outfile, open(FAILED_PATH, "a") as failed_file:
        with tqdm(total=total_prompts, desc="Generating Synthetic Prompts") as pbar:
            tasks = [
                generate_row(semaphore, adj_row, outfile, failed_file, pbar, id_counter)
                for adj_row in adjusted_rows
            ]
            await asyncio.gather(*tasks)

    print(f"✅ Done. Generated prompts saved to {OUTPUT_PATH}")
    print(f"   Failed entries logged to {FAILED_PATH}")

if __name__ == "__main__":
    asyncio.run(main())