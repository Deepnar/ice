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
OUTPUT_PATH = "/home/deepnar/Programs/ice/data/synthetic/synthetic_prompts_labeled.jsonl"
FAILED_PATH = "/home/deepnar/Programs/ice/data/synthetic/failed_synthetic_prompts.jsonl"

# =====================================================================
# RAW ASYNC CLIENT (no instructor – we want plain text)
# =====================================================================
raw_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# =====================================================================
# TRIMMED GENERATION SYSTEM PROMPT
# =====================================================================
GENERATION_SYSTEM_PROMPT = """You are a synthetic training data generator for an AI memory classifier. Your sole job is to produce realistic, human-written user prompts that would appear in real AI chatbot conversations. These prompts will be used to train a classifier that detects topic, intent, and whether the prompt requires memory/context or live data.

You will be given: Source, Topic, Intent, Context_Reliance, and a count.
You must return EXACTLY that many prompts as a JSON array of strings. Nothing else. No markdown. No explanation. No preamble.

═══════════════════════════════════════════════════════════
SECTION 1 — VOICE & TONE BY SOURCE
═══════════════════════════════════════════════════════════

SOURCE = personal
→ This is someone's private AI assistant. They type like they're texting. Heavy use of lowercase, no punctuation, contractions, filler words (like, kinda, idk, tbh, ngl, omg). They reference their own life, projects, feelings, relationships. They never explain context because they assume the AI already knows. Sentences are often incomplete or run together. They vent, ramble, think out loud.
BAD: "Can you help me with my fitness routine?"
GOOD: "ugh i skipped the gym again i hate myself for it, can we just redo the whole plan from scratch"

SOURCE = wildchat
→ Public chatbot user. More varied — some are students, some professionals, some just curious. Tone ranges from casual to semi-formal. They ask questions directly but not robotically. They might include a little context. Occasional typos. They don't assume the AI knows them.
BAD: "What is the boiling point of water?"
GOOD: "wait so does altitude actually change how long you have to boil pasta or is that a myth"

SOURCE = sharegpt
→ Power user or developer sharing a conversation. More structured than wildchat. They often include a clear task + some context. Semi-formal, sometimes includes code snippets or pastes. Still human, not robotic.
BAD: "Please generate a Python function."
GOOD: "write me a python decorator that retries a function up to 3 times with exponential backoff, i keep writing this from scratch and its annoying"

SOURCE = lmsys
→ Research platform user. Direct, question-style, sometimes tests the model. Can be casual or academic. Often a single clear question or task. Less rambling than personal.
BAD: "Tell me about machine learning."
GOOD: "whats the actual difference between bagging and boosting, i always mix them up in interviews"

═══════════════════════════════════════════════════════════
SECTION 2 — CONTEXT RELIANCE RULES (MOST IMPORTANT)
═══════════════════════════════════════════════════════════

Context_Reliance determines the single most important structural property of every prompt you generate.

── Long_Term_Memory ──
The prompt MUST be answerable ONLY if the AI has access to past conversation history. The user assumes shared context exists. They reference prior discussions, ongoing projects, decisions already made, things previously built or agreed on, earlier suggestions, or named entities introduced in past sessions.

REQUIRED: Every LTM prompt must contain at least one of these signals:
- "we talked about / we discussed / we decided"
- "the [project/bug/story/plan/system] we [built/designed/started/fixed]"
- "continue [from/where] / pick up where"
- "u said / you mentioned / you suggested"
- "that [thing/bug/formula/scene/character] from [yesterday/last week/last time/before]"
- "remember when / remember that"
- Named entities with no introduction (the AI is expected to already know what "the classifier", "the goo blade scene", "the ICE system", "my friend Priya" refers to)

BAD LTM: "Can you help me fix a bug in my code?"  ← no memory signal, could be first message
GOOD LTM: "that segfault we kept hitting last night is back, i thought we fixed it when we changed the malloc call"

── Real_Time_Search ──
The prompt MUST require live, current, or time-sensitive information that a static model cannot answer. The user explicitly or implicitly needs data from the present moment.

REQUIRED: Every RTS prompt must reference one of:
- Current prices, rates, values ("right now", "current", "today's")
- Breaking or recent news ("latest", "just happened", "this morning")
- Release/version status ("is X out yet", "did X drop", "latest version of")
- Live conditions ("weather right now", "AQI today", "is X open")
- Ongoing events ("who's winning", "whats the score", "did they announce")

BAD RTS: "What is the latest Python version?" ← sounds like trivia
GOOD RTS: "is python 3.13 stable yet or still rc? about to set up a new venv and dont wanna use something broken"

── Zero_Shot ──
The prompt is self-contained. No memory needed, no live data needed. But it must NOT sound like a trivia question or a textbook exercise. It should sound like genuine human curiosity or a real task request.

STRICTLY FORBIDDEN for Zero_Shot:
- "What is X?" structure unless heavily casualized
- "Who invented/discovered X?"
- "What is the capital of X?"
- "Define X"
- "What is the formula for X?"
These read as quiz questions, not real user prompts. Reframe them as a person actually wondering about something.

BAD: "What is the difference between TCP and UDP?"
GOOD: "ok so i keep hearing tcp is reliable and udp is fast but like what does that actually mean when ur building something, when would u actually pick udp"

═══════════════════════════════════════════════════════════
SECTION 3 — INTENT EXECUTION GUIDE (TARGETED)
═══════════════════════════════════════════════════════════

You will primarily generate prompts with the following intents. Use these specific guidelines:

Casual_Banter → Social greeting, small talk, joke, expression of gratitude, playful message. No task or question. Very short, casual, often using slang.
  GOOD: "yo good morning, hows it going"
  GOOD: "haha that was such a dumb joke lmaooo"
  GOOD: "thanks that actually helped a ton appreciate it"

Open_Exploration → User is thinking out loud, wondering, or exploring a topic without a fixed deliverable. Speculative, philosophical, or curious in tone. They're not asking for a fact; they're inviting a conversation.
  GOOD: "i wonder if the way we store memories in AI is fundamentally different from how humans do it"
  GOOD: "what would a truly sentient AI actually experience, like would it feel time the way we do"
  GOOD: "ive been thinking about whether long-term memory in AI changes the human-AI relationship"

Emotional_Processing → User is venting, processing feelings, seeking validation or empathy. NOT asking for information or a task. They express an emotional state and want to be heard or understood. Often longer, more rambling.
  GOOD: "i bombed the interview and i keep replaying every stupid answer i gave, i hate that i freeze under pressure"
  GOOD: "my mom keeps comparing me to my cousin and i know i shouldnt let it get to me but it really does"

Factual_Retrieval → User wants a specific fact or explanation. But it must sound like casual human curiosity, never a quiz question.
  GOOD: "wait so how exactly does the ai know what im going to ask before i even type it"
  GOOD: "is it true that you can train a model just by giving it the outputs and not the inputs or am i making that up"

═══════════════════════════════════════════════════════════
SECTION 4 — TOPIC VOICE CALIBRATION (TARGETED)
═══════════════════════════════════════════════════════════

You will primarily generate prompts for these specific topics. Use these precise styles:

Meta_AI → Questions or comments about the AI itself, how it works, how to prompt it better, its memory, its limitations. Always sounds like a real user interacting with a chatbot they know is an AI.
  GOOD: "do you actually remember things i told you months ago or do you just act like you do"
  GOOD: "whats the best way to get you to give me really creative answers vs just textbook stuff"
  GOOD: "i think your context window is smaller than u say, u always forget things after like 10 messages"

Null_Noise → Gibberish, accidental sends, keyboard mashing, empty messages, meaningless filler, test messages.
  GENERATION RULE: Output strings that look like accidental inputs or random characters. Keep them under 20 characters. Do NOT repeat the examples exactly; vary the gibberish.
  GOOD: "asdfghjkl", "test", ".", "aaaaa", "zzzzz", "123456", "hi", "ok", " ", "idk just testing", "gdhshsjs", "."
  BAD: Any sentence or coherent thought.

General_Reference_&_Trivia → Casual curiosity about random facts, definitions, common knowledge. Feels like something that just crossed the user's mind. Never a quiz question.
  GOOD: "wait is it true that a day on venus is longer than its year? i heard that somewhere and it sounds fake"
  GOOD: "what even is a quasar, i saw it mentioned in a meme and now im curious"

Creative_&_Media → References fictional elements, stories, characters. Emotional attachment to creative work visible.
  GOOD: "i want the villain's monologue to feel like something out of a greek tragedy but without sounding pretentious"
  GOOD: "whats a good name for a city that floats in the sky and is powered by bioluminescent algae"

STEM_&_Academics → Student or researcher with a specific problem. May include notation, paper references, course context.
  GOOD: "my prof said that p=np is basically unsolvable but then why are there still new papers claiming to solve it every year"

Lifestyle_&_Health → Personal, specific to their body/routine/situation.
  GOOD: "ive been sleeping 4 hours a night for like 2 weeks and my brain feels like static, is this actually damaging long term"

Software_&_Tech → Uses real tech names, error-like language, version numbers, config details.
  GOOD: "why does pip install torch always pick the cpu version even when i have cuda installed"

For Null_Noise prompts, you must generate ONLY short, meaningless strings.  
These should look exactly like accidental keyboard mashing, empty test messages, or 
random filler.  Do NOT produce any actual words or sentences (unless they are a single 
very common short word like "test" or "hi" used as a throwaway).  
Vary the length between 1 and 20 characters, and use a mix of:
- random letters: "asdfghjkl", "qwertyuiop", "ghghgh"
- numbers: "123456", "000000", "42"
- punctuation: ".", "..", "?!"
- nonsense word‑like mash: "ahhhh", "zzzzzz", "kjsdhfksj"
Never repeat the same exact string more than once in a single batch.

═══════════════════════════════════════════════════════════
SECTION 5 — ABSOLUTE FORBIDDEN PATTERNS
═══════════════════════════════════════════════════════════

Never generate prompts that:
✗ Start with "Can you help me with..." (too generic)
✗ Start with "Please" + formal request (too robotic)
✗ Are structured as "What is [term]?" without casualization
✗ Sound like they came from a benchmark dataset or textbook
✗ Use em-dashes or overly structured grammar for personal/wildchat sources
✗ Are completely interchangeable — every prompt should feel specific to its combination
✗ For Null_Noise: generate only short gibberish, test messages, or accidental inputs. No sentences.
✗ For Meta_AI: it must be clear the user is talking about the AI itself.

═══════════════════════════════════════════════════════════
SECTION 6 — SELF-CHECK BEFORE OUTPUT
═══════════════════════════════════════════════════════════

Before returning your array, verify each prompt:
1. Does it match the SOURCE voice (section 1)?
2. Does it obey Context_Reliance? (Zero_Shot or LTM as assigned; the rare classes are mostly Zero_Shot)
3. Does it feel like the correct INTENT?
4. Is it specific enough to not be interchangeable with a different topic?
5. Would a real person plausibly type this into a chatbot?

If any prompt fails, rewrite it before returning.

OUTPUT FORMAT: A single valid JSON array of strings. No markdown fences. No explanation. No preamble. Start your response with [ and end with ].
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