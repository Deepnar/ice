#!/usr/bin/env python3
"""Pile B batch 2 — long/rambling prompts, and the RARE signal combinations.

Batch 1 was mostly short one-liners. Real traffic is not: people paste, ramble,
change their mind mid-sentence, and stack three questions into one message. A
classifier trained only on terse prompts learns that length itself predicts the
label.

This batch also targets combinations the corpus barely has (counts over the
33,723 settled rows):

    High_Complexity + Live_Info                        51
    Live_Info + Temporal_Recall                        40
    Memory + Live_Info + Temporal_Recall               35
    High_Complexity + Temporal_Recall                  25
    High_Complexity + Live_Info + Memory               18
    all four signals                                    6
    High_Complexity + Live_Info + Temporal_Recall       2

Those are thin for the same censoring reason as the rest: they need web search
AND persistent memory to be worth asking, and the corpus environment had neither.

Some rows carry an attached context block — those are the hard cases where the
referent IS visible, so the correct answer is that memory is NOT needed.
"""
import sys

sys.path.insert(0, "/home/deepnar/Programs/ice/scripts/classifier/pipeline")

from authored import add, make_row, report  # noqa: E402

SW, CR, BF = "Software_&_Tech", "Creative_&_Media", "Business_&_Finance"
LH, SR, ST = "Lifestyle_&_Health", "Social_&_Relationships", "STEM_&_Academics"
AP, MA, GR = "Admin_&_Productivity", "Meta_AI", "General_Reference_&_Trivia"
WC = "World_&_Current_Events"
NM, TR, LI, HC = "Needs_Memory", "Temporal_Recall", "Needs_Live_Info", "High_Complexity"

ROWS = [
    # ── ALL FOUR SIGNALS (corpus has 6) ────────────────────────────────────
    ("ok so ages back we mapped out that whole migration plan for the storage layer, "
     "and i think i was leaning towards the option that kept the old ids around. can you "
     "dig that up, check whether the library we picked has shipped anything since then "
     "that changes the tradeoff, and then work out whether it still makes sense or whether "
     "i should just start over", [SW], ["Strategic_Planning", "Analysis_&_Summarization"], [NM, TR, LI, HC]),
    ("i want to revisit the investment thing we discussed months ago, pull up what my risk "
     "tolerance was back then, check where those funds are trading now, and tell me honestly "
     "whether the reasoning i used still holds or if i was being naive",
     [BF], ["Analysis_&_Summarization", "Decision_Making"], [NM, TR, LI, HC]),
    ("remember the stack i settled on for the side project last year? check if any of those "
     "are deprecated now and if so figure out what the least painful migration path would be "
     "given everything i told you about my constraints",
     [SW], ["Strategic_Planning"], [NM, TR, LI, HC]),

    # ── HC + Live + Temporal (corpus has 2) ────────────────────────────────
    ("how has the whole local llm situation changed since early this year, and given where "
     "things are heading whats the smart thing to bet on for the next 6 months",
     [SW], ["Analysis_&_Summarization", "Strategic_Planning"], [TR, LI, HC]),
    ("compare where ai regulation was a year ago to where its at now and reason through what "
     "that trajectory implies for small developers", [WC], ["Analysis_&_Summarization"], [TR, LI, HC]),

    # ── HC + Live + Memory (corpus has 18) ─────────────────────────────────
    ("given the hardware i told you i have, whats currently the best model i could actually "
     "run, and walk me through the tradeoffs properly not just a list",
     [SW], ["Analysis_&_Summarization", "Decision_Making"], [NM, LI, HC]),
    ("based on my situation that ive explained, and whats happening in the job market right "
     "now, should i be switching or staying put. think it through",
     [BF], ["Decision_Making"], [NM, LI, HC]),

    # ── HC + Temporal (corpus has 25) ──────────────────────────────────────
    ("trace how my architecture thinking evolved from the beginning to now and tell me if "
     "theres a pattern in the mistakes im making", [SW], ["Analysis_&_Summarization"], [NM, TR, HC]),
    ("what were the actual turning points in this project, the moments where a decision "
     "changed everything downstream", [SW], ["Analysis_&_Summarization"], [NM, TR, HC]),

    # ── Live + Temporal (corpus has 40) ────────────────────────────────────
    ("whats changed with that framework since the last major release", [SW], ["Factual_Retrieval"], [LI, TR]),
    ("has the price of that gpu moved much over the past few months", [SW], ["Factual_Retrieval"], [LI, TR]),
    ("hows the situation there developed since last week", [WC], ["Factual_Retrieval"], [LI, TR]),

    # ── Memory + Live + Temporal (corpus has 35) ───────────────────────────
    ("that api i integrated a while back, has their pricing changed since i set it up",
     [SW], ["Factual_Retrieval"], [NM, LI, TR]),
    ("the role i applied for months ago, is that posting still up or did they fill it",
     [BF], ["Factual_Retrieval"], [NM, LI, TR]),

    # ── long rambling, single signal ───────────────────────────────────────
    ("ok so this is gonna be long sorry. basically i've been going back and forth on whether "
     "to keep the whole thing as one service or split it. we talked about this before and i "
     "think you said something about the deployment overhead not being worth it at my scale "
     "but i cant remember if that was for this project or the other one. anyway can you find "
     "what we said and tell me if it applies here",
     [SW], ["Factual_Retrieval", "Decision_Making"], [NM]),
    ("so i've been writing the second act and its just not landing, the pacing feels off "
     "compared to the first one and i keep rewriting the same three scenes over and over. "
     "you know the structure i was going for, is the problem the structure itself or am i "
     "just too close to it right now",
     [CR], ["Analysis_&_Summarization", "Emotional_Processing"], [NM, HC]),
    ("right so heres the situation, ive got about four months of runway left, the product "
     "kind of works but nobody's paying, and i keep telling myself one more feature will do "
     "it. given everything ive told you about what i actually want out of this, be straight "
     "with me about whether i should keep going",
     [BF], ["Decision_Making", "Emotional_Processing"], [NM, HC]),
    ("i pasted you my config last time and you spotted the issue immediately but ive since "
     "changed a bunch of things and now its broken differently, the error mentions something "
     "about a missing key during startup but only sometimes which is the annoying part",
     [SW], ["Troubleshooting"], [NM]),

    # ── sporadic / stream of consciousness ─────────────────────────────────
    ("wait no. not that one. the other approach we talked about", [SW], ["Factual_Retrieval"], [NM]),
    ("hmm ok but what about... actually never mind, whats the thing i said about caching",
     [SW], ["Factual_Retrieval"], [NM]),
    ("ugh forget it. actually no wait — can you just tell me what my original plan was",
     [SW], ["Factual_Retrieval"], [NM]),
    ("ok so, like, the thing is?? i keep forgetting what we agreed on. can u check",
     [SW], ["Factual_Retrieval"], [NM]),
    ("nvm figured it out. actually one more thing, did i ever tell you why i picked postgres",
     [SW], ["Factual_Retrieval"], [NM]),

    # ── multiple questions stacked in one message ──────────────────────────
    ("three things. one, whats the current version of that lib. two, does it break the api i "
     "was using. three, if it does can you write the migration",
     [SW], ["Factual_Retrieval", "Code_Change"], [NM, LI]),
    ("can you remind me what the schema looked like, and also is there a newer approach people "
     "use now, and honestly is the whole design overcomplicated",
     [SW], ["Factual_Retrieval", "Analysis_&_Summarization"], [NM, LI, HC]),

    # ── context ATTACHED: referent visible, so memory NOT needed ───────────
    ("so which of those two should i go with",
     [SW], ["Decision_Making"], [],
     "User: i'm deciding between sqlite and postgres for the local store\n\n"
     "Assistant: Both work here. SQLite is zero-config and file-based, ideal if the store "
     "stays single-process. Postgres gives you pgvector, concurrent writers, and real "
     "indexes — worth it if retrieval grows."),
    ("ok do the second one",
     [SW], ["Code_Change"], [],
     "User: how should i handle the retries\n\nAssistant: Two options: a simple fixed-delay "
     "loop, or exponential backoff with jitter. The second is more code but survives a "
     "thundering herd."),
    ("why did you say that about the indexes",
     [SW], ["Factual_Retrieval"], [],
     "User: is postgres overkill for this\n\nAssistant: Not if you need vector search — "
     "pgvector's HNSW indexes are the reason most people move off SQLite for this workload."),
    ("expand on the third point",
     [SW], ["Analysis_&_Summarization"], [],
     "User: whats wrong with my current approach\n\nAssistant: Three things: the cache is "
     "unbounded, the retry has no ceiling, and the error swallows the original traceback."),

    # ── same shape, context attached, but STILL needs memory ───────────────
    ("does this match the convention we set for the other services",
     [SW], ["Codebase_Query"], [NM],
     "User: heres the new handler\n\nAssistant: Looks reasonable — it validates input, calls "
     "the service layer, and returns a typed response."),
    ("is this the approach i rejected last time",
     [SW], ["Factual_Retrieval"], [NM, TR],
     "User: what if we just poll every 30 seconds\n\nAssistant: That would work and it's "
     "simple, though it trades latency for load."),

    # ── hard negatives: long, sounds personal, actually self-contained ─────
    ("i need you to act as a senior engineer reviewing this code and give me feedback on "
     "style, correctness and performance. heres the function: def process(items): out = [] "
     "for i in items: out.append(i*2) return out",
     [SW], ["Analysis_&_Summarization"], []),
    ("write me a long detailed explanation of how transformers work, starting from attention "
     "and building up to a full block, assume i know linear algebra but nothing about ml",
     [ST], ["Generation"], [HC]),
    ("my character in this story im writing has a power that lets him rewind time by ten "
     "seconds. heres the full description of the magic system: [system uses lifeforce as "
     "currency, each rewind costs a year]. does this have any logical holes",
     [CR], ["Analysis_&_Summarization"], [HC]),
]

rows = []
for r in ROWS:
    if len(r) == 5:
        text, topic, intent, ctx, context = r
    else:
        text, topic, intent, ctx = r
        context = None
    rows.append(make_row(text, topic, intent, ctx,
                         note="long_and_rare_combinations", context_text=context))

n = add(rows)
print(f"added {n} rows (of {len(rows)} offered)\n")
report()
