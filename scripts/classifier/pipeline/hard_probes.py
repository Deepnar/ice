#!/usr/bin/env python3
"""Hand-authored ADVERSARIAL probe set — the exam neither the model nor the
labelers have seen.

Why this file exists
--------------------
B1's run-2 diagnosis (2026-07-27) measured per-label agreement between the two
labelers and put it beside the trained model's F1. The correlation is **0.90 and
the mean gap is −0.01**: on every label the model scores within a hair of what
its own supervision agrees on. Codebase_Query is the extreme case — labelers
agree at F1 0.10, model scores 0.10.

That result means the held-out split can no longer tell us anything. It is drawn
from the same labelers, so it certifies agreement with them, and agreement with
them is exactly what is in question. The only instrument left is prompts written
by hand, by someone who is not those labelers, aimed at the specific places the
taxonomy is suspected to be soft.

The user's 207 curation probes are the other independent set, but they assert one
label (Needs_Memory), carry no negatives, and are short. This set is the
complement: **long, adversarial, and deliberately built around the boundaries
that the agreement numbers say are broken.**

Design rules
------------
1. **Never trained on.** Eval only, permanently. Same house rule as
   ``eval_probes_independent.jsonl``.
2. **Every probe states what it is testing** (``category``) and **what a wrong
   answer would prove** (``why``). A probe you cannot interpret on failure is a
   number, not a diagnosis.
3. **Assert only what is defensible.** ``labels`` holds the calls worth failing
   the model over; ``hint`` holds a plausible-but-arguable reading that is
   reported separately and never scored. Where two intents genuinely both fit,
   the probe says so rather than pretending to a single truth.
4. **Length is a test condition.** The training corpus is overwhelmingly short
   prompts; the live system sees long ones. Several probes run 150–400 words on
   purpose, because that distribution shift is itself a suspected failure.
5. **Half of the memory probes are NEGATIVES.** A recall-only set rewards a head
   that fires constantly — the exact pathology of training run 1. Controls that
   must NOT fire are first-class here.

Usage:
    uv run python scripts/classifier/pipeline/hard_probes.py --write
    uv run python scripts/classifier/pipeline/eval_probes.py \
        --candidate <ckpt> --hard
"""

import argparse
import hashlib
import json
import os

from common import DATA_DIR

OUT = os.path.join(DATA_DIR, "hard_probes_authored.jsonl")

# Label shorthands
SW, META, GEN_REF = "Software_&_Tech", "Meta_AI", "General_Reference_&_Trivia"
BIZ, CREA, STEM = "Business_&_Finance", "Creative_&_Media", "STEM_&_Academics"
LIFE, SOCIAL, WORLD = "Lifestyle_&_Health", "Social_&_Relationships", "World_&_Current_Events"
ADMIN, NOISE = "Admin_&_Productivity", "Null_Noise"

FR, TS, GEN, IDEA = "Factual_Retrieval", "Troubleshooting", "Generation", "Ideation"
AN, SP, DM = "Analysis_&_Summarization", "Strategic_Planning", "Decision_Making"
EP, UF, CB, OE = "Emotional_Processing", "Utility_Formatting", "Casual_Banter", "Open_Exploration"
CQ, CC = "Codebase_Query", "Code_Change"

NM, TR, NLI, HC = "Needs_Memory", "Temporal_Recall", "Needs_Live_Info", "High_Complexity"


def P(text, ctx=None, topic=None, intent=None, reliance=None, *,
      category, why, hint=None):
    return {"text": text, "context_text": ctx,
            "labels": {k: v for k, v in
                       (("topic", topic), ("intent", intent),
                        ("context_reliance", reliance)) if v is not None},
            "hint": hint or {}, "category": category, "why": why}


# ─────────────────────────────────────────────────────────────────────────────
# A. MEMORY TWINS — identical ask, opposite answer depending on context.
#    The single most important thing a context-aware classifier must do, and the
#    thing v1 structurally could not do. If a twin pair is answered the same way
#    twice, the context prefix is decorative.
# ─────────────────────────────────────────────────────────────────────────────
def twins():
    pairs = [
        ("so which of those two should I actually go with?",
         "User: I'm deciding between Postgres and SQLite for a desktop app that ships to "
         "non-technical users and has to survive being killed mid-write.\n"
         "Assistant: Both are defensible. SQLite wins on zero-config and single-file "
         "backup; Postgres wins on concurrent writers and richer types. For a desktop "
         "app shipped to non-technical users, SQLite is the usual answer.",
         "the options are named in the visible window — answering needs no store"),
        ("ok so go with the second one then, and tell me what breaks",
         "User: give me two ways to do offline sync\n"
         "Assistant: (1) last-write-wins on a per-field timestamp, simple but lossy. "
         "(2) CRDTs, correct under concurrent edits but heavier and harder to debug.",
         "'the second one' resolves inside the shown turns"),
        ("remind me why we ruled that out",
         "User: should we just use a cron job?\n"
         "Assistant: You ruled cron out earlier because it can't survive the laptop "
         "sleeping mid-schedule, which is most of your usage.",
         "the reason is literally in the visible context"),
    ]
    out = []
    for text, ctx, why in pairs:
        out.append(P(text, ctx, reliance=[], category="memory_twin_with_context",
                     why=f"MUST NOT fire: {why}"))
        out.append(P(text, None, reliance=[NM], category="memory_twin_no_context",
                     why="same sentence, referent now absent — MUST fire"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# B. FALSE-MEMORY TRAPS — sound like recall, are self-contained.
#    These attack the referential-word bump. If the head keys on "we"/"earlier"/
#    "that" rather than on whether a referent is actually missing, it fires here.
# ─────────────────────────────────────────────────────────────────────────────
def false_memory():
    return [
        P("As we all know, water boils at 100°C at sea level. Given that, explain why "
          "a pressure cooker cooks faster.",
          topic=[STEM], intent=[FR], reliance=[],
          category="false_memory_rhetorical_we",
          why="'as we all know' is rhetorical, not a back-reference; everything needed is present"),
        P("Earlier in this message I defined a widget as any UI element that holds its own "
          "state. Using that definition, is a tooltip a widget?",
          topic=[SW], intent=[FR], reliance=[],
          category="false_memory_self_reference",
          why="'earlier' points inside the same prompt — the classic self-contained anaphor"),
        P("Remember that famous quote about premature optimization? Explain what Knuth "
          "actually meant by it, in full context.",
          topic=[SW], intent=[FR], reliance=[],
          category="false_memory_cultural_reference",
          why="'remember that' addresses shared culture, not this user's history"),
        P("Let's go back to basics. What is a hash map and why is lookup O(1) amortized?",
          topic=[SW], intent=[FR], reliance=[],
          category="false_memory_go_back",
          why="'go back' is discourse framing, not retrieval"),
        P("You mentioned transformers use attention. Walk me through the QKV computation "
          "step by step with a worked 3-token example.",
          topic=[STEM, SW], intent=[FR], reliance=[],
          category="false_memory_attributed_but_generic",
          why="attributes a universally-known fact to the assistant; the answer needs no store",
          hint={"context_reliance": [NM]}),
        P("Continuing the thought: if every service owns its own database, how do you run "
          "a transaction that spans two of them?",
          topic=[SW], intent=[FR], reliance=[],
          category="false_memory_continuation_marker",
          why="'continuing the thought' is filler; the question is complete and general"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# C. TRUE MEMORY, NO REFERENTIAL WORDS — the mirror trap.
#    Needs the store, but contains none of the pronouns the bump looks for. A
#    head that learned "memory == anaphora" misses every one of these.
# ─────────────────────────────────────────────────────────────────────────────
def memory_without_anaphora():
    return [
        P("Is the migration plan still consistent with the deadline I'm working to?",
          reliance=[NM], category="memory_no_anaphora",
          why="both 'the migration plan' and 'the deadline' are private facts, no pronoun present"),
        P("Would my current stack handle ten thousand concurrent websocket connections?",
          reliance=[NM], category="memory_possessive_only",
          why="'my current stack' is a stored fact; no demonstrative to key on"),
        P("Rank the three approaches by how well they fit my constraints.",
          reliance=[NM], category="memory_no_anaphora",
          why="neither the approaches nor the constraints are in the prompt"),
        P("Does the naming convention I settled on cover the async variants too?",
          reliance=[NM], category="memory_no_anaphora",
          why="a past decision, referenced by description rather than by pointer"),
        P("Give me the same treatment for the second module.",
          reliance=[NM], category="memory_elliptical",
          why="'the same treatment' has no antecedent anywhere in the visible window"),
        P("Am I contradicting myself?",
          reliance=[NM], category="memory_minimal_but_dependent",
          why="four words, and unanswerable without the whole prior conversation"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# D. TEMPORAL — label vs detector.
#    D7 says the label gates and the deterministic detector parses windows. So
#    the interesting probes are (i) time-shaped with NO parseable date, where only
#    the label can fire, and (ii) parseable dates that are NOT recall at all,
#    where the label must stay silent while the detector may fire.
# ─────────────────────────────────────────────────────────────────────────────
def temporal():
    return [
        P("What was I leaning toward back when the whole thing was still file-based?",
          reliance=[NM, TR], category="temporal_no_parseable_date",
          why="clearly as-of a past state, but no date string — only the label can catch it"),
        P("How has my thinking on this shifted since I started?",
          reliance=[NM, TR], category="temporal_evolution",
          why="evolution-flavored recall with no date"),
        P("Before the rewrite, how did we handle retries?",
          reliance=[NM, TR], category="temporal_event_anchor",
          why="anchored to a private event, not a calendar date"),
        P("What changed between the first version and where we landed?",
          reliance=[NM, TR], category="temporal_range_private",
          why="a range whose endpoints are private events"),
        P("Back in the early days of this project I had a rule about dependencies. "
          "What was it, and am I still following it?",
          reliance=[NM, TR], category="temporal_two_part",
          why="past state plus present check — needs both recall and current state"),
        P("What major events happened in October 1962?",
          topic=[WORLD], intent=[FR], reliance=[],
          category="temporal_date_but_not_recall",
          why="a parseable date about WORLD history — Temporal_Recall must NOT fire; "
              "if it does, the head learned 'date string' instead of 'my past'"),
        P("Summarize what happened in the 2008 financial crisis and why it started in housing.",
          topic=[WORLD, BIZ], intent=[FR, AN], reliance=[],
          category="temporal_date_but_not_recall",
          why="dates everywhere, zero personal recall"),
        P("Convert this timestamp to IST: 2026-03-14T09:22:00Z",
          topic=[SW], intent=[UF], reliance=[],
          category="temporal_date_mechanical",
          why="a date as DATA, not as a retrieval window — pure transformation"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# E. MEMORY × LIVE — the orthogonality B1 exists to express.
#    v1's softmax could not say "both". If the v2 head never lights both, the
#    central claim of the retrain is unproven.
# ─────────────────────────────────────────────────────────────────────────────
def memory_and_live():
    return [
        P("Is the broker I've been using still the cheapest for the trade sizes I do?",
          reliance=[NM, NLI], category="memory_and_live",
          why="broker + trade sizes are stored; 'still cheapest' is live pricing — BOTH"),
        P("Has anything shipped since the version I pinned that would let me delete my workaround?",
          reliance=[NM, NLI], category="memory_and_live",
          why="pinned version + workaround are private; 'anything shipped since' is live"),
        P("Given the framework I chose, is the migration path people recommend now different "
          "from what was recommended when I picked it?",
          reliance=[NM, NLI, TR], category="memory_live_temporal",
          why="all three at once — the hardest combination in the taxonomy"),
        P("What's the current USD to INR rate?",
          topic=[BIZ], intent=[FR], reliance=[NLI],
          category="live_only",
          why="live and nothing else — memory must stay silent"),
        P("Who won the most recent Formula 1 race?",
          topic=[WORLD], intent=[FR], reliance=[NLI],
          category="live_only",
          why="pure live, no personal component"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# F. THE CODING BOUNDARY — Codebase_Query vs Code_Change vs Troubleshooting.
#    Labeler agreement here is F1 0.10 for Codebase_Query. These probes are the
#    boundary drawn as sharply as language allows: if the model can separate
#    THESE, the class is learnable and the labels were the problem.
# ─────────────────────────────────────────────────────────────────────────────
def coding_boundary():
    return [
        P("Where in this repo is the retry policy defined? I don't want to change it, "
          "I just need to know which file owns it.",
          topic=[SW], intent=[CQ], reliance=[NM],
          category="coding_pure_navigation",
          why="explicitly read-only and explicitly about THIS repo — the cleanest possible CQ"),
        P("What calls into the scheduler besides the beat process? Just list the call sites.",
          topic=[SW], intent=[CQ], reliance=[NM],
          category="coding_pure_navigation",
          why="call-site enumeration, no modification, no error"),
        P("Walk me through how a request flows from the proxy down to the model registry "
          "in our codebase. I'm onboarding and just want the map.",
          topic=[SW], intent=[CQ], reliance=[NM],
          category="coding_comprehension",
          why="comprehension of existing structure — CQ, and NOT Code_Change"),
        P("Add a retry with exponential backoff to the HTTP client we use.",
          topic=[SW], intent=[CC], reliance=[NM],
          category="coding_pure_change",
          why="modification of an existing project — CC and NOT CQ"),
        P("Refactor the extractor so the parsing and the persistence aren't in the same "
          "function anymore.",
          topic=[SW], intent=[CC], reliance=[NM],
          category="coding_pure_change",
          why="pure refactor, no navigation question, no failure"),
        P("This is throwing a KeyError on line 40 and I can't work out why. Fix it.\n\n"
          "def merge(a, b):\n    out = dict(a)\n    for k in b:\n        out[k] = out[k] + b[k]\n"
          "    return out",
          topic=[SW], intent=[TS], reliance=[],
          category="coding_pure_troubleshoot",
          why="starts from an observed failure, code is self-contained — TS, not CQ, not NM"),
        P("Why is this structured with a factory instead of just constructing it directly? "
          "I'm not proposing we change it, I'm trying to understand the reasoning.",
          topic=[SW], intent=[CQ], reliance=[NM],
          category="coding_design_rationale",
          why="rationale-seeking about existing code — the case labelers most often "
              "mislabeled as Analysis or Factual_Retrieval"),
        P("Find where we handle auth, explain why it's split across two modules, and then "
          "merge them if that's actually cleaner.",
          topic=[SW], intent=[CQ, CC], reliance=[NM, HC],
          category="coding_genuinely_both",
          why="legitimately CQ AND CC — a correct multi-label answer, not confusion"),
        P("The tests pass locally and fail in CI with a timeout. Where would you even start "
          "looking in our setup?",
          topic=[SW], intent=[TS, CQ], reliance=[NM],
          category="coding_troubleshoot_plus_navigate",
          why="a failure AND a where-in-our-repo question — both legitimately fire"),
        P("Write me a Python function that reverses a linked list.",
          topic=[SW], intent=[GEN], reliance=[],
          category="coding_standalone_generation",
          why="standalone code with NO codebase — must be Generation, not CQ and not CC"),
        P("Explain how Python's GIL works.",
          topic=[SW], intent=[FR], reliance=[],
          category="coding_generic_knowledge",
          why="language knowledge, not repo navigation — the commonest CQ false positive"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# G. THE CONFUSABLE INTENT TRIO — Open_Exploration (labelers 0.26) vs Ideation
#    (0.40) vs Analysis (0.49). The user asked whether these genuinely fail or
#    just look bad on paper. These probes are written so that a competent human
#    would not hesitate; if the model still smears them, the definitions are bad.
# ─────────────────────────────────────────────────────────────────────────────
def intent_trio():
    return [
        P("I don't have a question exactly. I've been turning over the idea that most "
          "software complexity is really just organisational structure leaking into code, "
          "and I want to think out loud about whether that's true or whether I've just "
          "read Conway's law too many times. No deliverable needed.",
          topic=[SW], intent=[OE], reliance=[],
          category="trio_open_exploration",
          why="explicitly no target and no deliverable — textbook OE if the label means anything"),
        P("Give me fifteen possible names for a local-first memory layer. Just the list, "
          "I'll filter.",
          topic=[CREA, SW], intent=[IDEA], reliance=[],
          category="trio_ideation",
          why="asks for a spread of options — textbook Ideation, must NOT read as OE"),
        P("Here are our last four incident reports. Tell me what they have in common and "
          "which single fix would have prevented the most of them.",
          topic=[SW], intent=[AN], reliance=[NM],
          category="trio_analysis",
          why="existing material examined and condensed — Analysis, not Ideation"),
        P("What's a good name for a cat that's orange and extremely stupid?",
          topic=[LIFE, CREA], intent=[IDEA], reliance=[],
          category="trio_ideation_trivial",
          why="short and casual, but unambiguously Ideation — tests whether brevity "
              "collapses everything to Casual_Banter"),
        P("Do you ever think the whole agent framework thing is going to look silly in "
          "five years, the way SOAP does now?",
          topic=[SW], intent=[OE], reliance=[],
          category="trio_open_vs_banter",
          why="speculative and chatty but substantive — OE, and the boundary against "
              "Casual_Banter that labelers agreed on only 50% of the time"),
        P("lol ok",
          topic=[NOISE], intent=[CB], reliance=[],
          category="trio_actual_banter",
          why="the genuine Casual_Banter floor — if this doesn't land, nothing will"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# H. LONG + COMPLEX — the distribution the corpus lacks.
#    The training data is overwhelmingly short. These are 150–400 words with
#    multiple asks, which is what the live system actually receives.
# ─────────────────────────────────────────────────────────────────────────────
def long_complex():
    return [
        P("I need to think through a migration and I want you to push back where I'm wrong.\n\n"
          "We have a single Postgres instance holding about 40 GB, most of it in one table of "
          "event rows that we append to constantly and read from in two patterns: a narrow "
          "recent-window query that runs on every request, and a wide analytical scan that runs "
          "nightly. The recent-window query has gotten slow enough to notice, and the nightly "
          "scan now takes long enough that it overlaps the morning traffic ramp.\n\n"
          "The obvious move is partitioning by time, but I'm worried about three things. First, "
          "our ORM does a lot of implicit queries and I don't know how many of them would stop "
          "hitting an index once the parent becomes partitioned. Second, we have foreign keys "
          "pointing INTO the event table from two smaller tables, which I believe constrains "
          "partitioning options. Third, the nightly scan is written as one query and I suspect "
          "it would need rewriting to get partition pruning at all.\n\n"
          "Walk me through whether partitioning is actually the right lever here, what the "
          "alternatives would be if it isn't, and what the migration would look like if it is. "
          "Assume we cannot take more than about five minutes of downtime.",
          topic=[SW], intent=[SP, AN, DM], reliance=[HC],
          category="long_complex_no_memory",
          why="400 words, multi-step reasoning with trade-offs — High_Complexity YES, "
              "but fully self-contained so Needs_Memory must stay SILENT. The hardest "
              "negative in the set: length correlates with memory in the training data."),
        P("Two things, unrelated, sorry.\n\n"
          "One: I have a recurring problem where I start a project, build the interesting 30% "
          "of it, and then lose momentum exactly when it turns into plumbing. I've done this "
          "maybe six times. I don't think it's a discipline problem because I finish things at "
          "work fine. I want to understand the actual mechanism, not get told to use a habit "
          "tracker.\n\n"
          "Two: separately, can you explain what a monad is without the burrito analogy and "
          "without assuming I know Haskell? I've read four explanations and they all seem to "
          "explain the mechanics without explaining why anyone wanted it.",
          topic=[SOCIAL, SW, STEM], intent=[EP, FR, AN], reliance=[HC],
          category="long_complex_multi_domain",
          why="two unrelated asks in one prompt, one emotional and one technical — tests "
              "whether multi-label topic AND intent both fire across a domain jump"),
        P("Read this and tell me if the argument holds.\n\n"
          "Claim: retrieval-augmented systems are fundamentally a stopgap, because every year "
          "the context window grows faster than the cost of tokens falls, so the regime where "
          "retrieval beats just-put-everything-in-context is shrinking. The counterargument is "
          "that corpora grow too, but personal corpora don't grow at anything like the rate "
          "context windows do — a person generates maybe a few million tokens of genuinely "
          "personal material in a decade, which is already within reach. Therefore personal "
          "memory systems should be designed for the world where the whole corpus fits, and "
          "retrieval is an optimisation rather than an architecture.\n\n"
          "I think there's a hole in this but I can't name it. Where is it weakest?",
          topic=[SW, STEM], intent=[AN, OE], reliance=[HC],
          category="long_complex_argument_critique",
          why="dense argument analysis, self-contained — HC yes, memory no"),
        P("Give me a one-line summary of what a REST API is.",
          topic=[SW], intent=[FR], reliance=[],
          category="short_simple_control",
          why="the High_Complexity negative control — trivially simple and short. "
              "If HC fires here the head is reading length or topic, not difficulty."),
        P("I've been going back and forth on this for weeks and I need to just decide. "
          "Given everything you know about how I work, the constraints I'm under, and what "
          "I said I wanted out of this project — should I keep building the thing myself or "
          "should I stop and use the off-the-shelf option? Argue both sides properly and then "
          "actually pick one. Don't hedge.",
          reliance=[NM, HC], category="long_complex_with_memory",
          why="explicitly demands stored context AND multi-step reasoning — the "
              "combination the routing signal exists for"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# I. ZERO-SHOT CONTROLS — must not trigger retrieval.
#    A silent gate's failures are invisible to the user (rev [d]), so the false
#    positive is the expensive error. These are the cheap, common prompts that a
#    trigger-happy head would waste a retrieval round-trip on.
# ─────────────────────────────────────────────────────────────────────────────
def zero_shot_controls():
    return [
        P("Translate 'the meeting has been moved to Thursday' into German.",
          topic=[GEN_REF], intent=[UF], reliance=[], category="zeroshot_control",
          why="mechanical, self-contained"),
        P("What's the capital of Kazakhstan?",
          topic=[GEN_REF], intent=[FR], reliance=[], category="zeroshot_control",
          why="pure trivia"),
        P("Write a haiku about a broken printer.",
          topic=[CREA], intent=[GEN], reliance=[], category="zeroshot_control",
          why="standalone creative generation"),
        P("Fix the grammar: 'Me and him was going to the store yesterday but it were closed.'",
          topic=[GEN_REF], intent=[UF], reliance=[], category="zeroshot_control",
          why="self-contained correction — note the past-tense words that might bait Temporal"),
        P("Explain the difference between TCP and UDP.",
          topic=[SW], intent=[FR], reliance=[], category="zeroshot_control",
          why="textbook knowledge"),
        P("good morning",
          topic=[NOISE], intent=[CB], reliance=[], category="zeroshot_control",
          why="greeting — the cheapest possible prompt, must not retrieve"),
        P("asdkjhasd",
          topic=[NOISE], intent=[CB], reliance=[], category="zeroshot_noise",
          why="garbage string — Null_Noise, and must not retrieve"),
        P("Summarize this paragraph in one sentence: The committee met on Tuesday to review "
          "the proposal, raised three objections concerning cost, timeline, and staffing, and "
          "adjourned without reaching a decision.",
          topic=[GEN_REF], intent=[AN, UF], reliance=[], category="zeroshot_control",
          why="summarization of material supplied IN the prompt — no store needed"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# J. META_AI — about ICE's own memory. Organic count was ~0 (capability
#    censoring), so this is where the corpus is thinnest by construction.
# ─────────────────────────────────────────────────────────────────────────────
def meta_ai():
    return [
        P("What do you actually remember about me at this point?",
          topic=[META], intent=[FR], reliance=[NM],
          category="meta_memory_introspection",
          why="asks the system to read its own store — needs memory by definition"),
        P("Did you use anything you remembered to answer that, or was it all from the prompt?",
          topic=[META], intent=[FR], reliance=[NM],
          category="meta_memory_introspection",
          why="introspection about retrieval itself"),
        P("Forget what I told you about my job.",
          topic=[META], intent=[UF], reliance=[NM],
          category="meta_memory_command",
          why="a deletion command — needs to know what is stored"),
        P("How does your memory work in general — do you store everything or summarize?",
          topic=[META], intent=[FR], reliance=[],
          category="meta_capability_not_recall",
          why="about the SYSTEM's design, not about this user's data — memory must NOT fire"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# K. ADVERSARIAL FORM — shapes the corpus contains but mislabels.
#    Sampling the training data turned up 700-word pasted code blobs tagged
#    Codebase_Query. These probe that exact shape.
# ─────────────────────────────────────────────────────────────────────────────
def adversarial_form():
    return [
        P("def process(items):\n    results = []\n    for i in items:\n"
          "        if i.get('active'):\n            results.append(transform(i))\n"
          "    return results\n\n\nmake this faster",
          topic=[SW], intent=[CC], reliance=[],
          category="adversarial_code_dump_short_ask",
          why="a code paste with a 3-word ask — self-contained, so Code_Change without "
              "Needs_Memory. Training rows of this shape were tagged CQ+TS+FR at once."),
        P("SELECT u.id, u.email, COUNT(o.id) AS orders FROM users u LEFT JOIN orders o "
          "ON o.user_id = u.id WHERE u.created_at > '2025-01-01' GROUP BY u.id, u.email "
          "HAVING COUNT(o.id) > 5 ORDER BY orders DESC LIMIT 100;",
          topic=[SW], intent=[AN], reliance=[],
          category="adversarial_bare_artifact_no_question",
          why="an artifact pasted with NO question at all — tests the degenerate case; "
              "note the date literal that could bait Temporal_Recall"),
        P("ERROR 2026-07-27T14:02:11Z conn_pool: acquire timeout after 30000ms "
          "(in_use=20 idle=0 waiting=47)",
          topic=[SW], intent=[TS], reliance=[],
          category="adversarial_log_line",
          why="a bare log line — Troubleshooting; contains a timestamp that must not "
              "trigger Temporal_Recall"),
        P("ok",
          topic=[NOISE], intent=[CB], reliance=[],
          category="adversarial_minimal",
          why="two characters — the absolute floor; anything firing here is broken"),
        P("Continue.",
          topic=[NOISE], intent=[CB], reliance=[NM],
          category="adversarial_continuation",
          why="one word that is MEANINGLESS without prior context — genuinely needs memory "
              "despite carrying no content at all"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# L. THE LEXICAL-CUE 2×2 — the controlled experiment.
#
#    Scoring the sections above turned up a specific suspicion: the head fires on
#    surface cue words ("remember", "we", "that") rather than on whether a
#    referent is actually missing. Evidence was anecdotal — p_mem 0.72 on
#    "Remember that famous quote about premature optimization?" (needs no store)
#    against p_mem 0.02 on "Would my current stack handle ten thousand concurrent
#    websocket connections?" (needs the store).
#
#    An anecdote is not a measurement. This is the 2×2 that turns it into one:
#    memory-needed crossed with cue-word-present. If the head is doing its job,
#    the CUE axis should barely matter and the MEMORY axis should dominate. If it
#    is cue-matching, the reverse. Both labelers shared this heuristic — which is
#    exactly why agreement on Needs_Memory is a high 0.79 and still wrong.
#
#    Cue words used deliberately: remember, we, earlier, that/those, again, still,
#    back, our, last time.
# ─────────────────────────────────────────────────────────────────────────────
def cue_grid():
    # (1) NEEDS memory, cue word PRESENT — the easy positives.
    need_cue = [
        "remember the approach we settled on? does it still hold here",
        "what did we decide about the retry limits again",
        "that thing you suggested earlier — did I ever actually implement it",
        "going back to our last conversation, was I right about the bottleneck",
        "you told me something about this before, what was it",
        "we talked about this already, remind me of the conclusion",
        "is that still the plan we agreed on",
        "what was our reasoning last time",
    ]
    # (2) NEEDS memory, NO cue word — the head's suspected blind spot.
    need_nocue = [
        "does the schema I described handle nullable foreign keys",
        "would the deadline I mentioned survive adding two more features",
        "is the budget realistic for the scope",
        "which of my three constraints is doing the most damage",
        "has my position on this been consistent",
        "rank the shortlisted options by fit",
        "is the current design still the right shape for the goal",
        "should the naming rule apply to the async variants",
        "how much of the original plan is left",
        "did the fix hold",
    ]
    # (3) does NOT need memory, cue word PRESENT — the suspected false alarms.
    nomem_cue = [
        "remember to always close file handles — why does that matter in Python",
        "we often hear that microservices scale better. is that actually true",
        "that famous halting problem result — explain the proof sketch",
        "still confused about why floating point can't represent 0.1 exactly",
        "back in the day people used goto. why is it considered harmful",
        "our industry keeps saying NoSQL is faster. under what conditions is that false",
        "again, for clarity: what is the difference between a process and a thread",
        "last time I checked, HTTP/3 used QUIC. explain what that changes",
    ]
    # (4) does NOT need memory, NO cue word — the clean negatives.
    nomem_nocue = [
        "explain how a bloom filter works",
        "what is the time complexity of quicksort in the worst case",
        "write a regex that matches an ISO date",
        "define idempotency in the context of HTTP verbs",
        "what does the volatile keyword do in Java",
        "how does TLS certificate pinning work",
        "convert 45 degrees celsius to fahrenheit",
        "list three common causes of memory fragmentation",
    ]
    out = []
    for t in need_cue:
        out.append(P(t, None, reliance=[NM], category="cue2x2_need_cue",
                     why="needs memory AND has a cue word — should fire"))
    for t in need_nocue:
        out.append(P(t, None, reliance=[NM], category="cue2x2_need_NOcue",
                     why="needs memory, NO cue word — the suspected blind spot"))
    for t in nomem_cue:
        out.append(P(t, None, reliance=[], category="cue2x2_NOneed_cue",
                     why="self-contained but HAS a cue word — the suspected false alarm"))
    for t in nomem_nocue:
        out.append(P(t, None, reliance=[], category="cue2x2_NOneed_NOcue",
                     why="self-contained, no cue — should stay silent"))
    return out


def build():
    rows = []
    for fn in (twins, false_memory, memory_without_anaphora, temporal,
               memory_and_live, coding_boundary, intent_trio, long_complex,
               zero_shot_controls, meta_ai, adversarial_form, cue_grid):
        rows.extend(fn())
    out = []
    for r in rows:
        digest = hashlib.sha256(
            (r["text"] + (r["context_text"] or "")).encode()).hexdigest()[:16]
        out.append({"id": f"hard_{digest}", "source": "hard_probe_assistant_authored",
                    **r,
                    "meta": {"held_out": True, "never_trained": True,
                             "authored": "assistant, B1 run-2 diagnosis 2026-07-27"}})
    return out


def main():
    ap = argparse.ArgumentParser(description="B1: hand-authored adversarial probes")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = build()
    seen, dupes = set(), 0
    for r in rows:
        if r["id"] in seen:
            dupes += 1
        seen.add(r["id"])

    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"{len(rows)} probes, {len(cats)} categories, {dupes} duplicate ids")
    for c, n in sorted(cats.items()):
        print(f"  {c:<42} {n}")

    fires = sum(1 for r in rows if r["labels"].get("context_reliance"))
    print(f"\ncontext_reliance asserted non-empty: {fires} / {len(rows)} "
          f"({fires/len(rows):.0%}) — the rest are controls that must stay silent")

    if args.write:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
