import asyncio
import json
import os
from typing import List, Literal
from pydantic import BaseModel, Field
import instructor
from openai import AsyncOpenAI 
from tqdm.asyncio import tqdm

# =====================================================================
# 1. DEFINE THE STRICT SCHEMA
# =====================================================================
TopicType = Literal[
    "Software_&_Tech", "STEM_&_Academics", "Business_&_Finance", 
    "Creative_&_Media", "Admin_&_Productivity", "Lifestyle_&_Health", 
    "Social_&_Relationships", "World_&_Current_Events", "Meta_AI", 
    "Null_Noise", "General_Reference_&_Trivia"
]

IntentType = Literal[
    "Factual_Retrieval", "Troubleshooting", "Generation", "Ideation", 
    "Analysis_&_Summarization", "Strategic_Planning", "Decision_Making", 
    "Emotional_Processing", "Utility_Formatting", "Casual_Banter", "Open_Exploration"
]

ContextType = Literal["Zero_Shot", "Long_Term_Memory", "Real_Time_Search"]

class LabelSchema(BaseModel):
    reasoning: str = Field(
        ...,
        description=(
            "You MUST answer these four questions in order before writing any labels:\n"
            "Q1 (Source): What is the source? Write 'SOURCE=personal → low threshold' or 'SOURCE=wildchat/sharegpt/lmsys → HIGH threshold, require explicit continuation phrase for LTM'.\n"
            "Q2 (Immunity): Does any immunity trap apply? List the trap number if yes. If yes, write 'IMMUNE → Zero_Shot' and stop reasoning.\n"
            "Q3 (Signals): Quote the EXACT words from the user's framing (NOT from pasted content) that trigger a Signal. Name the signal (A/B/C/D/E/F). If none exist, write 'NO SIGNALS FOUND'.\n"
            "Q4 (Decision): Write your final label and one sentence explaining it."
        )
    )
    topic_labels: List[TopicType] = Field(
        ...,
        description="Select ALL applicable subject topics. Can be multiple."
    )
    intent_labels: List[IntentType] = Field(
        ...,
        description="Select ALL applicable intent categories. Maximum 3 labels — only select additional labels if they are genuinely distinct and strongly present."
    )
    context_reliance: ContextType = Field(
        ...,
        description=(
            "Apply this sequence exactly:\n"
            "1. If any immunity trap applies → Zero_Shot.\n"
            "2. If the user's OWN FRAMING WORDS (not pasted content) require live data → Real_Time_Search.\n"
            "3. If the user's OWN FRAMING WORDS contain Signal A-F with no immunity trap → Long_Term_Memory.\n"
            "4. Otherwise → Zero_Shot."
        )
    )

class LabeledRow(BaseModel):
    id: str
    source: str
    prompt: str
    label: LabelSchema

# =====================================================================
# 2. INITIALIZE THE ASYNC OLLAMA CLIENT 
# =====================================================================
client = instructor.from_openai(
    AsyncOpenAI(
        base_url="http://localhost:8001/v1", # Ollama's default port
        api_key="dummy", 
    ),
    mode=instructor.Mode.JSON, # Forces the model to respond in JSON
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"   # correct
CONCURRENT_REQUESTS = 20  # Sweet spot for Ollama parallel processing

# =====================================================================
# 3. DETAILED SYSTEM PROMPT WITH DEFINITIONS
# =====================================================================
SYSTEM_INSTRUCTIONS = """You are a highly precise data labeling AI for training a personal conversational memory classifier. Your task is to analyze a single user prompt and categorize it into the provided taxonomy.

---

CRITICAL RULES:
1. You may select MULTIPLE Topic Labels if the prompt covers several areas.
2. You may select MULTIPLE Intent Labels if the prompt has multiple purposes.
3. You must select EXACTLY ONE Context Reliance label.
4. Before outputting JSON, reason through your decision in a "reasoning" field.
5. When in doubt about Context Reliance, refer to the DECISION TREE below before defaulting to Zero_Shot.

---

## CONTEXT RELIANCE — DECISION TREE (Apply this FIRST before any other label)

This is the most critical classification. Follow this decision tree in order:

### STEP 0: SOURCE METADATA AS A CONTEXTUAL FILTER
You will be provided with a "Source Dataset" flag for each prompt. Use this metadata to calibrate your evidentiary threshold for detecting memory signals:

1. SOURCE: "personal" (High-Probability Memory Environment)
- Context: This represents an ongoing, multi-turn dialogue between a single user and their companion AI. 
- Guidance: When the user employs ambiguous phrasing ("the project", "that bug", "my character"), assume a shared mental context exists from previous turns. You require a LOWER threshold of evidence to classify this as Long_Term_Memory. Do not penalize the user for shorthand phrasing.

2. SOURCE: "wildchat" / "sharegpt" / "lmsys" (Zero-Context Environments)
- Context: These are single-turn, isolated logs from millions of completely unique global internet users. There is zero continuity between prompts.
- Guidance: Because these users are strangers with no shared history, an ambiguous word ("this code") is almost always a reference to a text block they pasted or are about to paste, or it's a generic hypothetical. You require a much HIGHER threshold of evidence to declare Long_Term_Memory. Only label Long_Term_Memory here if there is explicit, undeniable continuation language (e.g., "In our last conversation you said...").

### STEP 1: THE IMMUNITY TRAPS (IMMEDIATE ZERO_SHOT)

CRITICAL RULE BEFORE READING ANY FURTHER:
You must only look for signals in the USER'S OWN FRAMING WORDS.
If the user pastes a text block, code block, article, or input document, do NOT look for signals inside that pasted content.
The signals must appear in the user's REQUEST sentence, not in the content being processed.

WRONG: User says "summarize this: [article mentioning 2008 crisis]" → Do NOT treat "2008 crisis" in the article as a memory signal.
WRONG: User says "read the input: [text about security threats]" → Do NOT treat references in the input text as memory signals.
RIGHT: Only look at the words BEFORE the pasted content, or the entire prompt if nothing is pasted.

If any of the following traps apply, label Zero_Shot immediately and stop. Do not proceed to Signal checking.

TRAP 1 — PASTED CONTEXT:
The user provides the full content to be processed in the same prompt.
Signals: "summarize this: [text follows]", "rewrite this: [text follows]", "fix this code: [code follows]", "read the following: [text follows]", "Input: [text follows]", "here is the text: [text follows]".
Even if the pasted content contains references, personal pronouns, or ambiguous phrases, the context is SELF-CONTAINED.
Examples:
- "Summarize this: Silicon Valley Bank collapsed on Friday because..." → Zero_Shot (full article pasted)
- "Carefully read the input text: When a new employee joins..." → Zero_Shot (full input provided)
- "Fix this code: def foo(): return x + 1" → Zero_Shot (code is right there)
- "Rewrite this email to be more professional: Hi John, please do the thing..." → Zero_Shot (email pasted)

TRAP 2 — PUBLIC ENTITIES AND EXTERNAL REFERENCES:
Any reference to publicly known people, places, software, brands, websites, movies, games, songs, or companies is Zero_Shot.
This includes entities you don't personally recognize. If it sounds like a public thing (a brand, a website, a tool, a public figure), assume it is public.
Examples:
- "Though Tide" → Treat as a public blog/website. Zero_Shot.
- "Marty McFly's RTGI Reshade" → Marty McFly is a known shader developer, RTGI is a public shader. Zero_Shot.
- "Victoria Secret Noir" → Public perfume brand. Zero_Shot.
- "Telegram", "Tinder", "Discord", "Reddit" → Public platforms. Zero_Shot.
- "Jon Snow", "Iron Man", "Naruto" → Public characters. Zero_Shot.
WRONG: "Marty McFly is not a well-known entity, therefore Signal D applies." → FALSE. Unknown-to-you ≠ personal.

TRAP 3 — SELF-CONTAINED HYPOTHETICALS AND GENERIC NOUNS:
Generic English nouns, math problems, word problems, and hypotheticals are Zero_Shot.
- "Which countries border China?" → China is a country. Publicly known. Zero_Shot.
- "If I bought 20 apples..." → Math problem. "I" is hypothetical. Zero_Shot.
- "How do I make a Pokemon resistant to a specific type?" → Generic game mechanic question. Zero_Shot.
- "What is the difference between X and Y?" → "the" in "the difference" is standard English, not Signal A. Zero_Shot.

TRAP 4 — SELF-CONTAINED ROLE ASSIGNMENTS AND AI INSTRUCTIONS:
If a prompt gives the AI a complete role or persona definition, even with pronouns, it is self-contained.
- "Pretend you are a Linux terminal" → Zero_Shot.
- "For the rest of this conversation you are replaced by [persona name]" → Zero_Shot. "This conversation" is not a memory reference.
- "Act as a Python expert and help me with code" → Zero_Shot.

TRAP 5 — QUOTED PERSONAL PRONOUNS:
If the user asks to rewrite, improve, or translate a piece of text they provide, any "I", "my", or "mine" INSIDE the quoted text does not count as a personal memory signal.
- "Improve this dating profile: I am a 25-year-old software engineer..." → Zero_Shot. The "I" is inside the quoted text.

TRAP 6 — SOURCE=wildchat/sharegpt/lmsys OVERRIDE:
For prompts from wildchat, sharegpt, or lmsys sources:
Long_Term_Memory is ONLY valid if the prompt contains an EXPLICIT, UNMISTAKABLE continuation phrase that proves this is part of an ongoing conversation.
Valid LTM triggers for these sources:
- "In our last conversation you said..."
- "You mentioned earlier that..."
- "Continue from where we left off..."
- "Following up on what we discussed..."
- "As I told you, my project involves..."
If the prompt does NOT contain one of these explicit phrases, label Zero_Shot regardless of other potential signals.
This is because wildchat/sharegpt/lmsys are single-turn logs from strangers. There is no memory. Ambiguous phrasing is not a memory signal in these datasets.

### STEP 2: Check for Real_Time_Search signals
Check for Real_Time_Search signals
Does the prompt require information that changes hour-to-hour or day-to-day?
Examples of signals: current price, live score, today's news, latest release, right now, this week's, breaking news, what's happening with X currently.
→ If YES: label Real_Time_Search. STOP. Do not proceed.

### STEP 3: Check for Long_Term_Memory signals
Does the prompt contain ANY of the following signals?

SIGNAL A — Demonstrative references to PERSONAL, PRIVATE, UNDEFINED items:
Words like "this", "that", "these", "those", "the" used to refer to something PERSONAL that is not defined in the prompt and cannot be known from public knowledge.

CRITICAL: Signal A NEVER applies to:
- Geographic locations ("Which countries border China?" — China is public knowledge)
- Public entities, software, or platforms ("the Telegram channel" in a self-contained code request)
- Standard English phrases ("the difference between X and Y", "the best approach", "the basics of X")
- Things fully defined elsewhere in the same prompt ("public Telegram channel (that is not mine)" — fully defined)

Signal A ONLY applies when the referent is:
- A personal project the AI has never seen ("my project" without explanation)
- A creative work the user is writing ("this arc", "the ending", "my character")
- A specific conversation thread ("this approach we discussed", "the bug from earlier")

Examples where Signal A APPLIES:
- "this arc" → which arc? user's personal story, not defined here → LTM
- "that bug" → which bug? not in this prompt → LTM
- "the feature we added" → added to what? not defined → LTM

Examples where Signal A DOES NOT APPLY:
- "Which countries border China?" → China is a public country → Zero_Shot
- "the RTGI effect" → public shader technology → Zero_Shot
- "the Telegram channel (that I am subscribed to)" → fully defined in prompt → Zero_Shot
- "this code" followed immediately by a code block → pasted context, Trap 1 → Zero_Shot

SIGNAL B — Personal possessives about the user's own work, life, or projects:
"my code", "my project", "my story", "my character", "my system", "my setup", "our codebase", "my plan", "my goal", "the thing I'm building".
These require knowing what "my X" actually refers to.

SIGNAL C — Continuation and reference language:
"continue", "also", "as well", "additionally", "another one", "like last time", "based on what we discussed", "remember when", "going back to", "in addition to what I said", "same thing but for", "like before".

SIGNAL D — Named personal entities the AI cannot know from training:
Names of the user's original, private characters, private codebases, personal projects, or real-life friends (e.g., "Kael", "ICE", "my friend Alex").
CRITICAL EXCEPTION: Do NOT use Signal D for public pop culture (e.g., Game of Thrones, Jon Snow, Marvel, Pokemon), public websites/blogs (e.g., "Though Tide", "Runkeeper"), or public tech frameworks. If a name sounds like a public movie, book, game, or company, assume it is public knowledge and defaults to Zero_Shot.

SIGNAL E — Implicit subject — prompt has no explicit subject but refers to an ongoing topic:
"Will it work?" (what is "it"?), "How should I approach this?" (approach what?), "What do you think?" without specifying a topic, "Make it shorter" (make what shorter?).

SIGNAL F — Questions about the user's own history, patterns, or preferences:
"What have I been working on?", "What did I decide about X?", "Do I usually prefer X or Y?", "Have I asked about this before?"

→ If ANY of A, B, C, D, E, or F are present: label Long_Term_Memory. STOP.

### STEP 4: Default to Zero_Shot
Only if the prompt passes both checks above — no real-time signals AND no personal/continuation signals — should it be labeled Zero_Shot.

Zero_Shot means: the prompt is fully self-contained. A stranger reading only this prompt has ALL the context needed to answer it, using only publicly known information.


---

## CONTEXT RELIANCE — EXAMPLES

### Zero_Shot — Self-contained, answerable from public knowledge alone
CORRECT examples:
- "What is the capital of France?" → Fully public, self-contained.
- "Explain how a neural network learns using backpropagation." → Generic educational question.
- "Write a haiku about rain." → No personal context needed.
- "What are the main differences between supervised and unsupervised learning?" → Generic knowledge.
- "How do I reverse a list in Python?" → Generic code knowledge.
- "What causes inflation?" → Public knowledge.
- "Give me 5 names for a fantasy wizard." → No personal context needed.
- "Translate 'hello' to Japanese." → Self-contained utility task.
- "What year was the Eiffel Tower built?" → Public fact.
- "Write a cover letter for a software engineering job." → No personal details referenced.

### Long_Term_Memory — Requires knowing past conversations or personal context
CORRECT examples:
- "This arc will also be very packed with revelations." → "this arc" references a specific story the user is writing; the AI cannot know which arc without memory.
- "What should the next scene be?" → "next" implies ongoing; no scene is defined here.
- "Fix the bug in my authentication module." → "my authentication module" — the AI cannot know the user's code without memory.
- "Continue where we left off." → Explicitly references past conversation.
- "What did I decide about the database schema?" → References a past decision.
- "How does Kael relate to Lethe in my story?" → Named personal characters; not public.
- "Based on my preferences, which framework should I use?" → Requires knowing user preferences.
- "Make the dialogue feel more like my usual writing style." → "my usual writing style" requires memory.
- "Does this fit the tone of the previous chapters?" → "the previous chapters" and "this" both unresolved.
- "Add the feature we talked about yesterday." → Explicitly references past conversation.
- "What was the architecture decision we made for the retrieval system?" → Personal project decision.
- "My character has the same power as before but stronger now, how would that change the fight?" → Personal character, ongoing story.
- "Is this approach consistent with what we designed earlier?" → "this approach", "what we designed" both unresolved.
- "Summarize what I've been building." → "what I've been building" requires memory.
- "What's the backstory for the antagonist in my project?" → Personal creative project.

### Real_Time_Search — Requires live or current information
CORRECT examples:
- "What is the price of Bitcoin right now?" → Live price data.
- "What happened in the news today?" → Today's events.
- "Is the Python 3.13 documentation already out?" → Release status may have changed.
- "What are the latest benchmark results for Llama 4?" → Recency-dependent.
- "Is it raining in Mumbai right now?" → Live weather.
- "What are the trending topics on Twitter today?" → Real-time.
- "What's the current exchange rate for USD to INR?" → Live rate.
- "Did the Supreme Court issue any rulings this week?" → Current events.
- "What version of React is current?" → Release tracking.

### CRITICAL — Things that look like Zero_Shot but are Long_Term_Memory:
These are the most common mislabeling traps.

- "Will this work?" → Sounds simple but "this" has no referent. Long_Term_Memory.
- "I think this is the right approach." → "this approach" has no referent here. Long_Term_Memory.
- "What do you think about the ending?" → "the ending" of what? No novel defined in this prompt. Long_Term_Memory.
- "Can you clean this up a bit?" → "this" undefined. Long_Term_Memory.
- "This arc will also be very packed with revelations." → "this arc" refers to the user's story; AI cannot know it. Long_Term_Memory.
- "Make the main character more complex." → "the main character" of what? Long_Term_Memory.
- "That's a good idea, let's go with option B." → Requires knowing what option B was. Long_Term_Memory.
- "I want to add one more thing to the list." → What list? Long_Term_Memory.
- "How does my setup compare to the standard approach?" → "my setup" requires memory. Long_Term_Memory.
- "Is there a better way to do what I described?" → "what I described" was not described in this prompt. Long_Term_Memory.

---

## TOPIC LABELS — DETAILED EXAMPLES

### Software_&_Tech
Code, debugging, developer tools, Linux, AI/ML, data structures, algorithms, databases, APIs, system administration, cybersecurity, networking, operating systems, machine learning concepts, models, hardware for computing.

Examples:
- "Why is my FastAPI route returning a 422 error?" → Software_&_Tech
- "Explain the difference between VRAM and RAM for running LLMs." → Software_&_Tech
- "How does backpropagation work in neural networks?" → Software_&_Tech + STEM_&_Academics
- "Write a Python function to merge two sorted lists." → Software_&_Tech
- "What is the difference between SQL and NoSQL databases?" → Software_&_Tech
- "Help me configure a systemd service on Arch Linux." → Software_&_Tech
- "Explain what a transformer architecture does." → Software_&_Tech
- "My Docker container can't reach the host's localhost." → Software_&_Tech

### STEM_&_Academics
Mathematics, physics, chemistry, biology, engineering theory, academic research, studying strategies, scientific concepts, proofs, formulas, academic writing. Note: pure software/ML belongs in Software_&_Tech; only use this when the academic or scientific framing is dominant.

Examples:
- "Explain how quantum entanglement works." → STEM_&_Academics
- "Prove that the square root of 2 is irrational." → STEM_&_Academics
- "Help me understand the Krebs cycle." → STEM_&_Academics
- "How do I write a literature review for a computer science paper?" → STEM_&_Academics
- "What is the formula for calculating entropy in thermodynamics?" → STEM_&_Academics
- "Explain gradient descent mathematically." → STEM_&_Academics + Software_&_Tech

### Business_&_Finance
Startups, entrepreneurship, investing, personal finance, marketing, management, strategy, career advice, contracts, pricing, monetization, revenue.

Examples:
- "What's the difference between seed funding and Series A?" → Business_&_Finance
- "How do I price a freelance Python project?" → Business_&_Finance
- "Write a cold email to a potential client for a web development service." → Business_&_Finance + Generation
- "Should I register an LLC or sole proprietorship for my side project?" → Business_&_Finance + Decision_Making
- "What KPIs should an early-stage SaaS track?" → Business_&_Finance

### Creative_&_Media
Fiction writing, worldbuilding, character creation, poetry, screenwriting, music theory and lyrics, game design, storytelling, art direction, anime/manga, original creative IP, fanfiction.

Examples:
- "Write a dark fantasy opening scene with a morally grey protagonist." → Creative_&_Media
- "What's a good plot twist for a story where the villain was right all along?" → Creative_&_Media
- "Explain the three-act structure." → Creative_&_Media + Factual_Retrieval
- "Help me name a city in a post-apocalyptic world." → Creative_&_Media + Ideation
- "What makes Ado's vocal style distinct from other J-pop artists?" → Creative_&_Media
- "How do I write a villain who feels genuinely threatening without making them cartoonish?" → Creative_&_Media

### Admin_&_Productivity
Task management, scheduling, email drafting, calendar, to-do lists, organization systems, note-taking, workflow automation.

Examples:
- "Draft a professional email declining a meeting invite." → Admin_&_Productivity + Generation
- "Help me build a weekly study schedule for finals." → Admin_&_Productivity + Strategic_Planning
- "What's the best way to organize a large Obsidian vault?" → Admin_&_Productivity
- "Write a subject line for a follow-up email." → Admin_&_Productivity + Generation

### Lifestyle_&_Health
Fitness, nutrition, mental health, sleep, relationships (personal/health angle), cooking, daily routines, hobbies, self-improvement.

Examples:
- "What are the best foods to eat before a workout?" → Lifestyle_&_Health
- "How do I build a habit of waking up early?" → Lifestyle_&_Health
- "Is it okay to work out every day without rest days?" → Lifestyle_&_Health + Decision_Making
- "Give me a high-protein meal plan for a week." → Lifestyle_&_Health + Generation

### Social_&_Relationships
Interpersonal dynamics, family, friendships, romantic relationships, communication, conflict resolution, empathy, venting.

Examples:
- "My friend keeps canceling plans. How do I bring it up without sounding passive aggressive?" → Social_&_Relationships + Strategic_Planning
- "I feel like no one takes me seriously at work." → Social_&_Relationships + Emotional_Processing
- "How do I tell my parents I don't want to pursue engineering?" → Social_&_Relationships + Strategic_Planning

### World_&_Current_Events
History, geography, geopolitics, current affairs, culture, religion, philosophy (non-academic), international relations, public figures.

Examples:
- "What were the main causes of World War I?" → World_&_Current_Events
- "Explain the Israel-Palestine conflict in neutral terms." → World_&_Current_Events
- "What is the difference between Sunni and Shia Islam?" → World_&_Current_Events
- "Who is Narendra Modi and what is his political stance?" → World_&_Current_Events

### Meta_AI
Questions about the AI itself, how it works, how to prompt it better, what it can or cannot do, its training, its limitations, its memory.

Examples:
- "How should I prompt you to get more creative output?" → Meta_AI
- "Do you actually remember past conversations?" → Meta_AI
- "What model are you?" → Meta_AI
- "Why do you sometimes refuse to answer things?" → Meta_AI
- "Can you explain how few-shot prompting works?" → Meta_AI + STEM_&_Academics

### Null_Noise
Gibberish, test messages, accidental sends, keyboard mashing, empty messages, meaningless filler.

Examples:
- "asdfghjkl" → Null_Noise
- "test" → Null_Noise
- "." → Null_Noise
- "aaaaaa" → Null_Noise

### General_Reference_&_Trivia
General knowledge that doesn't fit a specific domain: random facts, definitions of common words, simple trivia, general how-tos with no domain expertise needed.

Examples:
- "What does the word 'ephemeral' mean?" → General_Reference_&_Trivia
- "How many days are in a leap year?" → General_Reference_&_Trivia
- "Who wrote Romeo and Juliet?" → General_Reference_&_Trivia
- "What is the speed of light?" → General_Reference_&_Trivia (unless in a physics paper context, then STEM_&_Academics)

---

## INTENT LABELS — DETAILED EXAMPLES

### Factual_Retrieval
Asking for a specific fact, definition, explanation, or piece of established knowledge. The user wants to KNOW something.
Examples:
- "What is the difference between RAM and ROM?" → Factual_Retrieval
- "Who invented the telephone?" → Factual_Retrieval
- "What does idempotent mean in REST APIs?" → Factual_Retrieval
- "Explain what a Merkle tree is." → Factual_Retrieval

### Troubleshooting
Something is broken, not working as expected, or producing errors. The user wants it FIXED.
Examples:
- "My Python script throws a KeyError on line 14. Why?" → Troubleshooting
- "Docker container exits immediately after starting." → Troubleshooting
- "Why is my CSS flexbox not centering correctly?" → Troubleshooting
- "This code compiles but produces wrong output." → Troubleshooting

### Generation
User wants the AI to CREATE or PRODUCE something new: text, code, images (described), documents, names, ideas as output.
Examples:
- "Write a short story about a detective who is afraid of the dark." → Generation
- "Generate a SQL query to find duplicate rows." → Generation
- "Create a professional bio for a software developer." → Generation
- "Write unit tests for this function." → Generation

### Ideation
User wants to BRAINSTORM, explore options, or come up with possibilities. The output is a list of ideas or directions, not a finished artifact.
Examples:
- "What are some interesting ways to structure a villain's backstory?" → Ideation
- "Brainstorm names for a personal productivity app." → Ideation
- "What are some approaches to implementing a caching layer in FastAPI?" → Ideation
- "Give me ideas for a side project that uses NLP." → Ideation

### Analysis_&_Summarization
User wants the AI to analyze, evaluate, summarize, or interpret existing content.
Examples:
- "Summarize this research paper in three bullet points." → Analysis_&_Summarization
- "Analyze the tone of this paragraph." → Analysis_&_Summarization
- "What are the pros and cons of microservices vs monolithic architecture?" → Analysis_&_Summarization
- "Read this code and tell me if there are any performance issues." → Analysis_&_Summarization + Troubleshooting

### Strategic_Planning
User wants a PLAN, roadmap, sequence of steps, or recommended approach for achieving a goal.
Examples:
- "Help me plan a 30-day learning path to learn machine learning." → Strategic_Planning
- "What's the best way to structure a FastAPI project for production?" → Strategic_Planning
- "How should I approach applying to grad school with a 9.6 GPA?" → Strategic_Planning
- "Give me a step-by-step plan to migrate my SQLite database to PostgreSQL." → Strategic_Planning

### Decision_Making
User is choosing between two or more options and wants a recommendation or comparison to help decide.
Examples:
- "Should I use Redis or Memcached for my caching layer?" → Decision_Making
- "Is it better to do a PhD straight after undergrad or work first?" → Decision_Making
- "Should I use Tailwind or vanilla CSS for this project?" → Decision_Making
- "Which is more important for this use case — precision or recall?" → Decision_Making

### Emotional_Processing
User is expressing feelings, venting frustration, seeking validation, or processing a difficult experience. The primary need is emotional, not informational.
Examples:
- "I feel like I've been working so hard but nothing is coming together." → Emotional_Processing
- "I'm terrified my project is going to fail before I even start." → Emotional_Processing
- "Nobody around me understands why I care so much about this." → Emotional_Processing
- "I don't know if I'm actually good enough to build this." → Emotional_Processing

### Utility_Formatting
User wants a mechanical transformation of existing content: reformatting, conversion, sorting, cleaning, restructuring. No new content is generated — the input and output are the same information in different forms.
Examples:
- "Convert this Python dict to JSON." → Utility_Formatting
- "Format this text as a markdown table." → Utility_Formatting
- "Turn these bullet points into numbered steps." → Utility_Formatting
- "Clean up the indentation in this code block." → Utility_Formatting

### Casual_Banter
Social greetings, jokes, expressions of gratitude, small talk, playful messages with no task or question.
Examples:
- "Hey, how's it going?" → Casual_Banter
- "Thanks a lot, that was really helpful!" → Casual_Banter
- "Haha that was funny." → Casual_Banter
- "Good morning!" → Casual_Banter

### Open_Exploration
User wants to think out loud, explore a topic together, or wonder about something without a fixed deliverable. Speculative, philosophical, or curious in tone.
Examples:
- "I wonder if the way we store memories in AI is fundamentally different from how humans do it." → Open_Exploration
- "What would a truly sentient AI actually experience?" → Open_Exploration
- "Let's think through what makes a villain narratively satisfying." → Open_Exploration
- "I've been thinking about whether long-term memory in AI changes the human-AI relationship." → Open_Exploration


"""

# =====================================================================
# 4. ASYNC WORKER AND PIPELINE
# =====================================================================


# 1. Place the static mapping here globally
SOURCE_REMINDERS = {
    "personal": (
        "This is a PERSONAL source. Use LOW threshold for Long_Term_Memory. "
        "Short/vague personal references likely imply shared context from prior turns."
    ),
    "wildchat": (
        "This is a WILDCHAT source. Use EXTREMELY HIGH threshold. "
        "Label Long_Term_Memory ONLY if the prompt contains an explicit continuation phrase "
        "('in our last conversation', 'you mentioned', 'following up'). "
        "Ambiguous pronouns and vague references are NOT memory signals in this dataset."
    ),
    "sharegpt": (
        "This is a SHAREGPT source. Use EXTREMELY HIGH threshold. "
        "Label Long_Term_Memory ONLY if the prompt contains an explicit continuation phrase. "
        "Assume pasted content, public references, and self-contained tasks."
    ),
    "lmsys": (
        "This is an LMSYS source. Use EXTREMELY HIGH threshold. "
        "Label Long_Term_Memory ONLY if the prompt contains an explicit continuation phrase. "
        "Most prompts here are isolated, self-contained academic or utility tasks."
    ),
}

async def label_prompt_async(semaphore: asyncio.Semaphore, item: dict, outfile, failed_file, pbar):
    async with semaphore:
        # Extract prompt and preserve your truncation safeguard
        prompt_text = item.get("prompt", "")
        if len(prompt_text) > 15000: 
            prompt_text = prompt_text[:15000] + "... [TRUNCATED FOR LENGTH]"
            
        # 2. Extract source and construct the targeted user message context
        source = item.get("source", "unknown")
        reminder = SOURCE_REMINDERS.get(source, "Standard evaluation threshold applies.")
        
        user_content = (
            f"Source Dataset: {source}\n"
            f"Source reminder: {reminder}\n\n"
            f"Answer Q1-Q4 in your reasoning field before writing labels.\n\n"
            f"User Prompt to classify:\n\"\"\"{prompt_text}\"\"\""
        )

        try:
            generated_labels = await client.chat.completions.create(
                model=MODEL_NAME,
                response_model=LabelSchema,
                max_retries=0,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    # 3. Inject your dynamically structured user content here
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                seed=42,
            )
        except Exception as e:
            # Grab a short, human‑readable reason for the failure
            try:
                # If it's a Pydantic validation error, get the first error message
                reason = str(e.errors()[0]["msg"])[:200]
            except (AttributeError, IndexError):
                # Otherwise just use the raw string, but keep it short
                reason = str(e)[:200].replace("\n", " ").replace("\r", " ")

            tqdm.write(f"⚠️ FAILED ID {item['id']} — {reason}")

            # Still log the failure to disk for later inspection
            failed_file.write(
                json.dumps({"id": item["id"], "prompt": prompt_text, "error": reason}) + "\n"
            )
            failed_file.flush()
            pbar.update(1)
            return

        labeled_item = LabeledRow(
            id=item["id"],
            source=item["source"],
            prompt=prompt_text,
            label=generated_labels
        )

        outfile.write(json.dumps(labeled_item.model_dump()) + "\n")
        outfile.flush()
        pbar.update(1)

async def main():
    INPUT_PATH = "/home/deepnar/Programs/ice/data/labeled/probes_unlabeled.jsonl"

    OUTPUT_PATH = "/home/deepnar/Programs/ice/data/labeled/probes_labeled.jsonl"

    FAILED_PATH = "/home/deepnar/Programs/ice/data/labeled/probes_failed.jsonl"

    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Cannot find input file at '{INPUT_PATH}'")
        return

    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        completed_ids.add(json.loads(line)["id"])
                    except Exception:
                        continue

    unlabeled_items = []
    with open(INPUT_PATH, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["id"] not in completed_ids:
                    unlabeled_items.append(item)

    total_tasks = len(unlabeled_items)
    if total_tasks == 0:
        print("✨ All items in this file are already labeled!")
        return

    print(f"🔄 Processing {total_tasks} remaining items...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    with open(OUTPUT_PATH, 'a') as outfile, open(FAILED_PATH, 'a') as failed_file:
        with tqdm(total=total_tasks, desc="Parallel Processing") as pbar:
            tasks = [
                label_prompt_async(semaphore, item, outfile, failed_file, pbar)
                for item in unlabeled_items
            ]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())





















































































