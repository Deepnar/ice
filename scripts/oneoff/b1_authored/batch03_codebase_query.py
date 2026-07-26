#!/usr/bin/env python3
"""Pile B batch 3 — Codebase_Query, the label the corpus could not contain.

65 organic rows exist because the training conversations were website chats with
no repo access: "where is the retry logic in my project?" was a pointless
question, so it was never asked. Under MCP it becomes one of the commonest
things a developer says.

Boundary the labelers kept collapsing (they resolve ambiguity to Code_Change
almost every time, which is why these are authored rather than generated):

    Codebase_Query  — READ. Understand code that exists. Nothing changes.
    Code_Change     — WRITE. The codebase differs afterwards.
    Troubleshooting — starts from an observed FAILURE.
    Generation      — standalone code, no codebase attached.

Most rows carry Needs_Memory: answering "where does X live" requires the code
graph, which is outside the prompt and outside the visible window. A handful are
deliberately NOT Codebase_Query (generic language questions) so the model learns
the boundary rather than the keyword "where".
"""
import sys

sys.path.insert(0, "/home/deepnar/Programs/ice/scripts/classifier/pipeline")

from authored import add, make_row, report  # noqa: E402

SW = "Software_&_Tech"
NM, TR, LI, HC = "Needs_Memory", "Temporal_Recall", "Needs_Live_Info", "High_Complexity"
CQ, CC, TS, GEN, AN, FR = ("Codebase_Query", "Code_Change", "Troubleshooting",
                           "Generation", "Analysis_&_Summarization", "Factual_Retrieval")

# (text, intent(s), ctx) — topic is Software_&_Tech unless noted
Q = [
    # ── plain "where is X" ────────────────────────────────────────────────
    ("where is the retry logic", [CQ], [NM]),
    ("where do we handle auth", [CQ], [NM]),
    ("which file has the db connection setup", [CQ], [NM]),
    ("where's the config loaded from", [CQ], [NM]),
    ("point me to where the embeddings get created", [CQ], [NM]),
    ("what file is the scheduler in", [CQ], [NM]),
    ("where do i find the migration files", [CQ], [NM]),
    ("wheres the entry point for this thing", [CQ], [NM]),
    ("which module owns the cache", [CQ], [NM]),
    ("where is that constant defined", [CQ], [NM]),
    ("find me the class that does the parsing", [CQ], [NM]),
    ("where does the request first hit our code", [CQ], [NM]),
    ("locate the function that builds the prompt", [CQ], [NM]),
    ("what directory are the tests in again", [CQ], [NM]),
    ("show me where we set the timeout", [CQ], [NM]),

    # ── "how does X work" ─────────────────────────────────────────────────
    ("how does the retrieval actually work in this repo", [CQ], [NM]),
    ("explain how the worker picks up jobs", [CQ], [NM]),
    ("walk me through what happens when a request comes in", [CQ], [NM, HC]),
    ("how does the caching layer decide what to evict", [CQ], [NM]),
    ("whats the flow from user input to stored memory", [CQ], [NM, HC]),
    ("how do these two services talk to each other", [CQ], [NM]),
    ("explain the auth middleware to me like im new here", [CQ], [NM]),
    ("how is the config merged, env vs file", [CQ], [NM]),
    ("whats the lifecycle of one of these objects", [CQ], [NM]),
    ("how does the retry backoff actually compute the delay", [CQ], [NM]),
    ("can you explain what this decorator is doing in our codebase", [CQ], [NM]),
    ("how does the batching work under the hood here", [CQ], [NM]),

    # ── "what calls / what uses" (graph queries) ──────────────────────────
    ("what calls this function", [CQ], [NM]),
    ("who uses this helper", [CQ], [NM]),
    ("is this method used anywhere anymore", [CQ], [NM]),
    ("what depends on the storage module", [CQ], [NM]),
    ("if i change this signature what breaks", [CQ], [NM, HC]),
    ("what imports this file", [CQ], [NM]),
    ("are there other callers of this besides the api", [CQ], [NM]),
    ("whats downstream of this change", [CQ], [NM, HC]),
    ("does anything still reference the old table", [CQ], [NM]),
    ("trace every path that ends up writing to the db", [CQ], [NM, HC]),

    # ── "why is it like this" (design comprehension) ──────────────────────
    ("why is this split into two files", [CQ], [NM]),
    ("why do we have both of these functions, they look the same", [CQ], [NM]),
    ("whats the reason for the extra abstraction layer here", [CQ], [NM]),
    ("why is this done synchronously", [CQ], [NM]),
    ("is there a reason we dont just use the library directly", [CQ], [NM]),
    ("whats this weird workaround for", [CQ], [NM]),
    ("why does this catch the exception and swallow it", [CQ], [NM]),
    ("was there a reason we picked this pattern here", [CQ], [NM, TR]),

    # ── comprehension of state / structure ────────────────────────────────
    ("whats in the schema right now", [CQ], [NM]),
    ("how many workers do we actually have", [CQ], [NM]),
    ("list the endpoints we expose", [CQ], [NM]),
    ("whats the current test coverage situation", [CQ], [NM]),
    ("what are all the settings we read from env", [CQ], [NM]),
    ("give me an overview of the folder structure", [CQ], [NM]),
    ("summarise what this module does", [CQ, AN], [NM]),
    ("what are the main abstractions in this codebase", [CQ, AN], [NM, HC]),
    ("whats the biggest file in here and why is it so big", [CQ], [NM]),
    ("do we have anything that already does this", [CQ], [NM]),
    ("is there an existing helper for this or do i need to write one", [CQ], [NM]),
    ("what does this function return exactly", [CQ], [NM]),
    ("are there print statements left in this one", [CQ], [NM]),
    ("whats the default value for that param", [CQ], [NM]),

    # ── lazy / short / typo'd ─────────────────────────────────────────────
    ("wheres the retry thing", [CQ], [NM]),
    ("wht file was the parser in", [CQ], [NM]),
    ("hwo does this work again", [CQ], [NM]),
    ("that funciton, where is it", [CQ], [NM]),
    ("the config, where", [CQ], [NM]),
    ("which one of these actually runs", [CQ], [NM]),
    ("remind me what this does", [CQ], [NM]),
    ("wait where is that handled", [CQ], [NM]),

    # ── with a time dimension ─────────────────────────────────────────────
    ("when did we add this function", [CQ], [NM, TR]),
    ("has this file changed much recently", [CQ], [NM, TR]),
    ("what did this look like before the refactor", [CQ], [NM, TR]),
    ("who wrote this originally and why", [CQ], [NM, TR]),
    ("was this always structured this way", [CQ], [NM, TR]),

    # ── long / rambling comprehension ─────────────────────────────────────
    ("ok so im trying to understand the retrieval path properly. i can see theres a "
     "bm25 leg and a vector leg but im not clear on how the weights get chosen or "
     "where that happens. can you walk me through it end to end without changing "
     "anything, i just want to understand it first", [CQ], [NM, HC]),
    ("before i touch anything i want to know what im dealing with. whats the actual "
     "dependency situation between the workers and the api layer, and is there "
     "anywhere theyre coupled in a way thats gonna bite me", [CQ, AN], [NM, HC]),
    ("i inherited this and have no idea whats going on. give me the tour, start with "
     "whats most important to understand", [CQ], [NM, HC]),

    # ── BOTH read and write (legitimately multi-label) ────────────────────
    ("find where we validate the token and fix the expiry bug", [CQ, CC, TS], [NM]),
    ("show me the retry code then make it exponential", [CQ, CC], [NM]),
    ("whats the current implementation, and can you clean it up", [CQ, CC], [NM]),
    ("explain how this works then refactor it to be less nested", [CQ, CC], [NM, HC]),

    # ── boundary: Code_Change, NOT Codebase_Query ─────────────────────────
    ("add caching to the loader", [CC], [NM]),
    ("refactor this into a service class", [CC], [NM]),
    ("implement the delete endpoint", [CC], [NM]),
    ("migrate this to the new api", [CC], [NM, HC]),
    ("delete the dead code in there", [CC], [NM]),
    ("wire the new worker into the scheduler", [CC], [NM]),
    ("make the timeout configurable", [CC], [NM]),
    ("add a test for this function", [CC], [NM]),

    # ── boundary: Troubleshooting, NOT Codebase_Query ─────────────────────
    ("why am i getting a 422 on this endpoint", [TS], [NM]),
    ("this test hangs and i dont know why", [TS], [NM]),
    ("its throwing a keyerror on startup but only sometimes", [TS], [NM]),
    ("the worker died silently again", [TS], [NM]),

    # ── boundary: Generation, NOT Codebase_Query (no codebase) ────────────
    ("write a python function that reverses a linked list", [GEN], []),
    ("give me a regex that matches an email", [GEN], []),
    ("show me an example of a context manager", [GEN], []),
    ("whats the syntax for a dataclass again", [FR], []),

    # ── boundary: generic language question, looks like "where/how" ───────
    ("where does python look for imported modules", [FR], []),
    ("how does asyncio actually schedule coroutines", [FR], [HC]),
    ("what calls __init__ in python", [FR], []),
    ("where are pip packages installed by default", [FR], []),
    ("how does git store objects internally", [FR], [HC]),
    ("why is this pattern called dependency injection", [FR], []),
]

rows = [make_row(t, [SW], i, c, note="codebase_query") for t, i, c in Q]
n = add(rows)
print(f"added {n} rows (of {len(rows)} offered)\n")
report()
