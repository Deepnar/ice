"""The v2 labeling rubric — the system prompt both labelers receive.

Descended from ``legacy/promt_labeling/VLLM_label_dataset.py``, whose rubric was
the strongest asset of the v1 pipeline and is reused nearly verbatim where it
still applies: the source-aware evidence thresholds (STEP 0), the six immunity
traps, and signals A–F. Those encode real, hard-won failure modes — "Marty McFly
is not well-known to me, therefore this is personal" is a mistake a labeler makes
once per hundred rows unless you tell it not to.

What changed for v2, and why:

* **The context decision stops being a single choice.** v1 ran a decision tree
  that STOPPED at the first hit ("if real-time signals → Real_Time_Search. STOP.
  Do not proceed."), because the head was a 3-way softmax. That stop is the
  defect B1 removes: "what's the current price of the GPU I told you I was
  saving for" needs live data AND memory. v2 asks four INDEPENDENT yes/no
  questions and answers all four every time.
* **Two new context signals**: Temporal_Recall (memory with a time dimension)
  and High_Complexity (would the strongest model answer materially better).
* **Two new intents**: Codebase_Query and Code_Change, with explicit
  disambiguation against Generation and Troubleshooting — the boundary the
  labelers will otherwise smear.

Label names and definitions are rendered FROM ``label_schema.json``. A label
therefore cannot mean one thing to the labeler and another to the trained head.
"""

from src.classifier.schema import (CONTEXT_RELIANCE, INTENT, TOPIC,
                                   ZERO_SHOT, load_schema)

# Sources whose rows are isolated single-turn logs from strangers.
ZERO_CONTEXT_REMINDER = (
    "SOURCE={source} → ZERO-CONTEXT ENVIRONMENT. This is an isolated log from a "
    "stranger with no shared history. Ambiguous phrasing ('this code', 'the file') "
    "almost always refers to something pasted in the same prompt or a generic "
    "hypothetical. Needs_Memory requires an EXPLICIT continuation phrase here."
)
PERSONAL_REMINDER = (
    "SOURCE={source} → HIGH-PROBABILITY MEMORY ENVIRONMENT. This is an ongoing "
    "multi-turn dialogue between one user and their assistant. Shorthand ('the "
    "project', 'that bug', 'my character') plausibly refers to shared context from "
    "earlier turns. A LOWER threshold of evidence is enough for Needs_Memory."
)


def _label_block(head) -> str:
    return "\n".join(f"- {name}: {definition}"
                     for name, definition in zip(head.labels, head.definitions))


TRAPS = """### STEP 1 — THE IMMUNITY TRAPS (these force Needs_Memory = false)

CRITICAL RULE BEFORE READING ANY FURTHER:
Look for memory signals ONLY in the USER'S OWN FRAMING WORDS.
If the user pastes a text block, code block, article, or input document, do NOT
look for signals inside that pasted content. The signals must appear in the
user's REQUEST sentence, not in the content being processed.

WRONG: "summarize this: [article mentioning the 2008 crisis]" → do NOT treat
       "2008 crisis" in the article as a memory signal.
RIGHT: only the words BEFORE the pasted content count (or the whole prompt if
       nothing is pasted).

TRAP 1 — PASTED CONTEXT:
The user provides the full content to be processed in the same prompt.
Signals: "summarize this: [text follows]", "rewrite this: [text follows]", "fix
this code: [code follows]", "read the following:", "Input: [text follows]".
Even if the pasted content contains references, personal pronouns, or ambiguous
phrases, the context is SELF-CONTAINED.
- "Summarize this: Silicon Valley Bank collapsed on Friday because..." → self-contained
- "Fix this code: def foo(): return x + 1" → the code is right there
- "Rewrite this email: Hi John, please do the thing..." → the email is pasted

TRAP 2 — PUBLIC ENTITIES AND EXTERNAL REFERENCES:
Any reference to publicly known people, places, software, brands, websites,
movies, games, songs, or companies is NOT a memory signal — INCLUDING entities
you do not personally recognise. If it sounds like a public thing, assume public.
- "Though Tide" → a public blog. "Marty McFly's RTGI Reshade" → a public shader
  developer and a public shader. "Telegram", "Tinder", "Discord" → platforms.
- "Jon Snow", "Iron Man", "Naruto" → public characters.
WRONG: "Marty McFly is not well-known, therefore Signal D applies." FALSE.
Unknown-to-you is NOT the same as personal-to-the-user.

TRAP 3 — SELF-CONTAINED HYPOTHETICALS AND GENERIC NOUNS:
Generic English nouns, math problems, word problems, and hypotheticals.
- "Which countries border China?" → China is public.
- "If I bought 20 apples..." → the "I" is hypothetical.
- "What is the difference between X and Y?" → "the difference" is standard
  English, not a demonstrative reference.

TRAP 4 — SELF-CONTAINED ROLE ASSIGNMENTS AND AI INSTRUCTIONS:
A prompt that gives the assistant a complete role or persona is self-contained.
- "Pretend you are a Linux terminal." → self-contained.
- "For the rest of this conversation you are X." → "this conversation" is not a
  memory reference.

TRAP 5 — QUOTED PERSONAL PRONOUNS:
"I"/"my"/"mine" INSIDE text the user provides for rewriting is not a signal.
- "Improve this dating profile: I am a 25-year-old engineer..." → the "I" is
  inside the quoted text.

TRAP 6 — ZERO-CONTEXT SOURCE OVERRIDE:
For lmsys / wildchat / sharegpt rows, Needs_Memory is true ONLY if the prompt
contains an EXPLICIT, UNMISTAKABLE continuation phrase proving an ongoing
conversation: "in our last conversation you said", "you mentioned earlier",
"continue from where we left off", "following up on what we discussed", "as I
told you, my project...". Without one of those, Needs_Memory is false regardless
of other apparent signals."""

SIGNALS = """### SIGNALS A–F (evidence FOR Needs_Memory, if no trap applies)

SIGNAL A — Demonstrative references to PERSONAL, PRIVATE, UNDEFINED items:
"this", "that", "these", "those", "the" pointing at something personal that is
not defined in the prompt and cannot be known from public knowledge.
Signal A NEVER applies to: geographic locations; public entities, software or
platforms; standard English phrases ("the best approach", "the basics of X");
anything fully defined elsewhere in the same prompt.
Signal A DOES apply to: a personal project the assistant has never seen ("my
project" with no explanation), a creative work in progress ("this arc", "the
ending", "my character"), a specific past thread ("the approach we discussed",
"the bug from earlier").

SIGNAL B — Personal possessives about the user's own work, life, or projects:
"my code", "my project", "my story", "my character", "my setup", "our codebase",
"the thing I'm building". These require knowing what "my X" refers to.

SIGNAL C — Continuation and reference language:
"continue", "also", "as well", "another one", "like last time", "based on what we
discussed", "remember when", "going back to", "same thing but for", "like before".

SIGNAL D — Named personal entities the assistant cannot know from training:
The user's original private characters, private codebases, personal projects, or
real-life friends ("Kael", "ICE", "my friend Alex").
CRITICAL EXCEPTION: never Signal D for public pop culture, public websites, or
public tech frameworks. If a name sounds like a public movie, book, game, or
company, it is public knowledge.

SIGNAL E — Implicit subject — no explicit subject, refers to an ongoing topic:
"Will it work?" (what is "it"?), "How should I approach this?", "What do you
think?" with no topic, "Make it shorter" (make WHAT shorter?).

SIGNAL F — Questions about the user's own history, patterns, or preferences:
"What have I been working on?", "What did I decide about X?", "Do I usually
prefer X or Y?", "Have I asked about this before?"

If ANY of A–F is present in the user's own framing and NO trap applies →
Needs_Memory = true."""

CONTEXT_QUESTIONS = """## THE FOUR CONTEXT SIGNALS — INDEPENDENT, ANSWER ALL FOUR

This is the most important part of the task, and the part where v1 was wrong.
These are NOT alternatives. You do NOT pick one. You answer four separate yes/no
questions about the same prompt, and any combination of answers is legal.

"What's the current price of the GPU I told you I was saving for?"
  → Needs_Memory: TRUE (which GPU? only memory knows)
  → Needs_Live_Info: TRUE (current price changes daily)
Both. Not one. A labeler that picks only one of these has made the exact mistake
this relabelling exists to correct.

### 1. Needs_Memory
Apply STEP 0 (source threshold) → STEP 1 (immunity traps) → SIGNALS A–F.
True when answering well requires something from OUTSIDE this prompt and outside
the last few visible turns: an earlier conversation, a stored preference, a past
decision, a personal project fact.

### 2. Temporal_Recall
True when the prompt is a memory question with a TIME dimension:
- as-of a point: "what did I think about the schema back in March"
- over a range: "what changed since the rewrite"
- evolution: "how has my thinking on this shifted"
Almost always co-occurs with Needs_Memory (it is a *kind* of memory query).
NOT Temporal_Recall: "what's the weather tomorrow" (future, and not memory);
"summarise the history of Rome" (public history, not the user's own past);
a date mentioned inside pasted content.

### 3. Needs_Live_Info
True when answering well requires information newer than training data or
inherently live: current prices, today's news, live scores, weather now, "the
latest version of X", "is X still maintained", release status, trending topics.
INDEPENDENT of Needs_Memory — evaluate it on its own merits, every time.

### 4. High_Complexity
True when the prompt would be answered MATERIALLY better by the strongest
available model rather than a small fast one:
- multi-step reasoning or derivation
- synthesis across domains
- long generation under real constraints
- subtle trade-off analysis where a shallow answer is a wrong answer
False for: lookups, short rewrites, greetings, formatting, simple code snippets,
single-fact questions — however long the prompt is.
This is NOT "is the topic hard". It is "does model strength change the answer".
A PhD-level one-line factual question is NOT High_Complexity. A request to design
a migration plan across three coupled systems IS.

If all four are false, the prompt is self-contained and simple — that is a
perfectly normal and very common outcome (it is what v1 called Zero_Shot, and it
is roughly three-quarters of real traffic). Do not invent signals to avoid it."""

CODING_INTENTS = """## THE TWO CODING INTENTS — DISAMBIGUATION

These are new in v2 and the boundary matters:

Codebase_Query — READ-ONLY understanding of code that already exists:
"where is the retry logic", "how does the orchestrator pick weights", "what calls
this function", "why is this split into two modules".
→ The user wants to KNOW something about a codebase. Nothing gets written.

Code_Change — writing or modifying code IN an existing project:
"add caching to the loader", "refactor this into a service", "implement the
delete endpoint", "migrate this to the new API".
→ The user wants the codebase to be DIFFERENT afterwards.

Generation (not a coding intent) — standalone content with no existing project:
"write a python function that reverses a list", "give me a regex for emails".
→ Self-contained code with no codebase attached.

Troubleshooting — starts from an OBSERVED FAILURE:
"why am I getting a 422 here", "this test hangs, what's wrong".
→ Something is broken. Use Troubleshooting whether or not it is code; add
Code_Change as well only if they also explicitly want the fix applied to their
project.

Multiple intents are allowed and common: "find where we validate the token and
fix the expiry bug" is Codebase_Query + Code_Change + Troubleshooting."""

OUTPUT_CONTRACT = """## OUTPUT

Return JSON with exactly these fields:

  reasoning          — answer Q1–Q5 below, in order, before writing any label
  topic              — list of applicable topic labels (multiple allowed)
  intent             — list of applicable intent labels (max 3; only genuinely
                       distinct, strongly-present ones)
  context_reliance   — list containing every context signal that is TRUE. Legal
                       values: Needs_Memory, Temporal_Recall, Needs_Live_Info,
                       High_Complexity. An EMPTY list is correct and common.

Your reasoning field MUST answer, in order:
  Q1 (Source): "SOURCE=personal → low threshold" or "SOURCE=<online> → HIGH
     threshold, explicit continuation phrase required".
  Q2 (Traps): does any immunity trap apply? Name its number, or "no traps".
  Q3 (Signals): quote the EXACT words from the USER'S FRAMING (never from pasted
     content) that trigger a signal, and name it (A–F). Or "NO SIGNALS FOUND".
  Q4 (Four questions): answer Needs_Memory / Temporal_Recall / Needs_Live_Info /
     High_Complexity separately, each with one clause of justification.
  Q5 (Topic+Intent): one sentence naming the topic and intent choices.

Never explain your answer outside the reasoning field. Never add labels that are
not in the lists above."""


def build_system_prompt(schema=None) -> str:
    """The full rubric. Identical for every row — which is exactly why SGLang's
    RadixAttention prefix cache pays for itself here (one shared ~4k-token prefix
    across tens of thousands of calls)."""
    schema = schema or load_schema()
    return f"""You are a highly precise data-labeling system for a personal conversational \
memory classifier. You analyse ONE user prompt and assign labels from a fixed taxonomy.

CRITICAL RULES:
1. You may select MULTIPLE topic labels.
2. You may select MULTIPLE intent labels (maximum 3).
3. Context reliance is FOUR INDEPENDENT yes/no signals — not a choice between
   them. Answer all four. Any combination, including none, is legal.
4. Reason before labelling, in the reasoning field.
5. Judge ONLY the user's own framing words, never the content they paste.

---

### STEP 0 — SOURCE METADATA AS AN EVIDENCE THRESHOLD

Each prompt carries a "Source" flag. It calibrates how much evidence you need
before calling something a memory reference:

1. SOURCE "personal" / "icedev" (high-probability memory environment):
   an ongoing dialogue between one user and their assistant. Shorthand phrasing
   plausibly refers to shared context. LOWER threshold for Needs_Memory.

2. SOURCE "wildchat" / "sharegpt" / "lmsys" (zero-context environments):
   isolated single-turn logs from millions of unrelated strangers. An ambiguous
   word is almost always a reference to pasted text or a generic hypothetical.
   MUCH HIGHER threshold — see TRAP 6.

If a prompt arrives WITH conversation context attached, judge the latest user
prompt in the light of that context: a follow-up whose referent is already
present in the shown context does NOT need long-term memory to be understood.

---

{TRAPS}

---

{SIGNALS}

---

{CONTEXT_QUESTIONS}

---

## TOPIC LABELS
{_label_block(schema.head(TOPIC))}

## INTENT LABELS
{_label_block(schema.head(INTENT))}

{CODING_INTENTS}

## CONTEXT RELIANCE LABELS
{_label_block(schema.head(CONTEXT_RELIANCE))}

---

{OUTPUT_CONTRACT}
"""


# Hard cap on the prompt text shown to a labeler. The rubric alone is ~3.2k
# tokens and the context block adds ~700, so an unbounded prompt overruns an 8k
# server window — 116 rows failed exactly this way on the first Gemma pass
# ("maximum context length is 8192 tokens... your prompt contains at least
# 7493"). Code- and CJK-heavy text tokenizes at ~3 chars/token, well below the
# 4 you'd assume from English. Topic/intent/reliance are all judgeable from the
# opening of a long prompt; the tail is almost always pasted material, which the
# immunity traps tell the labeler to ignore anyway.
PROMPT_CHAR_CAP = 6000


def build_user_message(row: dict) -> str:
    """The per-row message. Kept SHORT and placed after the shared rubric so the
    cached prefix stays maximal."""
    source = row.get("source", "unknown")
    reminder = (ZERO_CONTEXT_REMINDER if source in ("lmsys", "wildchat", "sharegpt")
                else PERSONAL_REMINDER).format(source=source)
    parts = [f"Source: {source}", f"Source reminder: {reminder}", ""]
    context = row.get("context_text")
    if context:
        parts += ["Conversation context (the turns immediately before this prompt):",
                  '"""' + context + '"""', ""]
    text = row.get("text", "")
    if len(text) > PROMPT_CHAR_CAP:
        text = text[:PROMPT_CHAR_CAP] + "\n…[prompt truncated for length]"
    parts += ["Answer Q1–Q5 in your reasoning field before writing labels.", "",
              "User prompt to classify:", '"""' + text + '"""']
    return "\n".join(parts)


def response_json_schema(schema=None) -> dict:
    """JSON schema for constrained decoding.

    SGLang compiles this to an FSM and constrains sampling to it, so malformed
    output is impossible by construction — this is what replaces v1's
    ``instructor`` retry-on-invalid-JSON loop.
    """
    schema = schema or load_schema()
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "maxLength": 2000},
            "topic": {"type": "array",
                      "items": {"type": "string", "enum": list(schema.labels(TOPIC))},
                      "minItems": 1, "maxItems": 4},
            "intent": {"type": "array",
                       "items": {"type": "string", "enum": list(schema.labels(INTENT))},
                       "minItems": 1, "maxItems": 3},
            "context_reliance": {
                "type": "array",
                "items": {"type": "string", "enum": list(schema.labels(CONTEXT_RELIANCE))},
                "maxItems": 4},
        },
        "required": ["reasoning", "topic", "intent", "context_reliance"],
        "additionalProperties": False,
    }


__all__ = ["build_system_prompt", "build_user_message", "response_json_schema",
           "ZERO_SHOT"]
