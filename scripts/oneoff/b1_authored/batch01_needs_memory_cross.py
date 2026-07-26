#!/usr/bin/env python3
"""Pile B batch 1 — Needs_Memory ACROSS conversations (0 examples in the corpus).

Every one of these refers to something said in a DIFFERENT chat/session. That is
the case ICE exists to serve and the corpus has none of it, because in a website
chat the assistant cannot see other conversations so nobody ever phrases it this
way.

Written as a human types: lowercase starts, typos left in, no trailing
punctuation half the time, lengths from three words to a rambling sentence.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "Programs", "ice", "scripts", "classifier", "pipeline"))
sys.path.insert(0, "/home/deepnar/Programs/ice/scripts/classifier/pipeline")

from authored import add, make_row, report

SW = "Software_&_Tech"
CR = "Creative_&_Media"
BF = "Business_&_Finance"
LH = "Lifestyle_&_Health"
SR = "Social_&_Relationships"
ST = "STEM_&_Academics"
AP = "Admin_&_Productivity"
MA = "Meta_AI"
GR = "General_Reference_&_Trivia"
NM, TR, LI, HC = "Needs_Memory", "Temporal_Recall", "Needs_Live_Info", "High_Complexity"

# (text, topic, intent, ctx)
ROWS = [
    # --- plain cross-conversation recall, no time word -------------------
    ("in that other chat we settled on a name for the retry helper, what was it", [SW], ["Factual_Retrieval"], [NM]),
    ("whats the db schema we ended up with in our other convo", [SW], ["Factual_Retrieval"], [NM]),
    ("we talked about this in a different chat, whats the conclusion we reached", [SW], ["Factual_Retrieval"], [NM]),
    ("pull up what i said about the pricing model in the other conversation", [BF], ["Factual_Retrieval"], [NM]),
    ("i explained my whole setup in another chat, can you use that", [SW], ["Factual_Retrieval"], [NM]),
    ("you helped me with the auth flow in a separate convo, continue from there", [SW], ["Code_Change"], [NM]),
    ("remind me what we decided on for the character arc, different chat", [CR], ["Factual_Retrieval"], [NM]),
    ("what did i tell you my budget was, it was in another session", [BF], ["Factual_Retrieval"], [NM]),
    ("the outline i gave you before in the other thread, use that structure", [CR], ["Generation"], [NM]),
    ("i already described my project somewhere else, dont make me repeat it", [SW], ["Factual_Retrieval"], [NM]),
    ("we had a whole discussion on this, other chat, whats the summary", [SW], ["Analysis_&_Summarization"], [NM]),
    ("carry over the constraints from our previous conversation", [SW], ["Strategic_Planning"], [NM]),
    ("what was the reason i rejected the queue approach, we discussed it elsewhere", [SW], ["Factual_Retrieval"], [NM]),
    ("in one of our earlier chats i listed my requirements, what were they", [SW], ["Factual_Retrieval"], [NM]),
    ("you gave me a really good analogy for this in another convo, what was it", [ST], ["Factual_Retrieval"], [NM]),

    # --- with a time reference (Needs_Memory + Temporal_Recall) ----------
    ("what did we decide about the schema last month", [SW], ["Factual_Retrieval"], [NM, TR]),
    ("last week in a different chat i mentioned a library, which one was it", [SW], ["Factual_Retrieval"], [NM, TR]),
    ("couple weeks back we went over my deployment plan, whats changed since", [SW], ["Analysis_&_Summarization"], [NM, TR]),
    ("i had a whole plan for this back when we started, what happened to it", [SW], ["Factual_Retrieval"], [NM, TR]),
    ("how has my thinking on the storage layer changed over our chats", [SW], ["Analysis_&_Summarization"], [NM, TR]),
    ("before the rewrite what was i using for embeddings", [SW], ["Factual_Retrieval"], [NM, TR]),
    ("months ago i asked you about this same bug, did we fix it", [SW], ["Troubleshooting"], [NM, TR]),
    ("what was my original idea for the ending, before i changed it", [CR], ["Factual_Retrieval"], [NM, TR]),
    ("compare what i want now vs what i wanted when we first talked", [SW], ["Analysis_&_Summarization"], [NM, TR]),
    ("earlier on i was against using a queue, am i still", [SW], ["Analysis_&_Summarization"], [NM, TR]),
    ("what were my goals when i started this whole thing", [BF], ["Factual_Retrieval"], [NM, TR]),
    ("did i ever settle on a workout split, that was a while back", [LH], ["Factual_Retrieval"], [NM, TR]),
    ("a while ago you suggested three options and i picked one, which", [SW], ["Factual_Retrieval"], [NM, TR]),
    ("whats my track record on finishing these side projects", [BF], ["Analysis_&_Summarization"], [NM, TR]),
    ("how did the argument with my manager end up, we talked about it before", [SR], ["Factual_Retrieval"], [NM, TR]),

    # --- questions about the user's own patterns / preferences (signal F) --
    ("do i usually prefer tabs or spaces, you should know by now", [SW], ["Factual_Retrieval"], [NM]),
    ("whats my usual writing style, match it", [CR], ["Generation"], [NM]),
    ("based on everything i've told you, what kind of dev am i", [SW], ["Analysis_&_Summarization"], [NM, HC]),
    ("what topics do i keep coming back to", [MA], ["Analysis_&_Summarization"], [NM]),
    ("have i asked you about this before", [MA], ["Factual_Retrieval"], [NM]),
    ("what do you actually know about my project so far", [MA], ["Factual_Retrieval"], [NM]),
    ("summarise everything you know about me", [MA], ["Analysis_&_Summarization"], [NM]),
    ("am i repeating myself, feels like ive asked this", [MA], ["Factual_Retrieval"], [NM]),
    ("what have i been working on lately", [SW], ["Factual_Retrieval"], [NM, TR]),
    ("do i normally go for the simple option or the clever one", [SW], ["Analysis_&_Summarization"], [NM]),
    ("whats a decision i made that you think i should revisit", [SW], ["Analysis_&_Summarization"], [NM, HC]),
    ("list the projects ive mentioned to you", [SW], ["Factual_Retrieval"], [NM]),
    ("what did i say i was bad at", [SR], ["Factual_Retrieval"], [NM]),
    ("do i tend to over engineer stuff", [SW], ["Analysis_&_Summarization"], [NM]),

    # --- short, lazy, referential (the hardest realistic case) ------------
    ("whats that thing i mentioned", [SW], ["Factual_Retrieval"], [NM]),
    ("the one from before", [SW], ["Factual_Retrieval"], [NM]),
    ("same as last time please", [SW], ["Generation"], [NM]),
    ("use my usual format", [AP], ["Utility_Formatting"], [NM]),
    ("like we did for the other project", [SW], ["Code_Change"], [NM]),
    ("what was that library i liked", [SW], ["Factual_Retrieval"], [NM]),
    ("continue where we left off", [SW], ["Generation"], [NM]),
    ("finish the thing from yesterday", [SW], ["Code_Change"], [NM, TR]),
    ("go back to the earlier version", [CR], ["Code_Change"], [NM]),
    ("that character i made up, whats his backstory again", [CR], ["Factual_Retrieval"], [NM]),
    ("the approach we picked, remind me why", [SW], ["Factual_Retrieval"], [NM]),
    ("whats the name i chose for it", [CR], ["Factual_Retrieval"], [NM]),

    # --- cross-conversation AND needs live info (both signals) ------------
    ("that gpu i said i was saving for, is it still that price", [SW], ["Factual_Retrieval"], [NM, LI]),
    ("the library you recommended me before, has it been updated", [SW], ["Factual_Retrieval"], [NM, LI]),
    ("is the framework i picked still maintained", [SW], ["Factual_Retrieval"], [NM, LI]),
    ("check if theres a newer version of the thing i was using", [SW], ["Factual_Retrieval"], [NM, LI]),
    ("that stock i mentioned wanting to buy, hows it doing now", [BF], ["Factual_Retrieval"], [NM, LI]),
    ("the conference i said i wanted to go to, when is it this year", [BF], ["Factual_Retrieval"], [NM, LI]),
    ("did the bug i reported upstream ever get fixed", [SW], ["Troubleshooting"], [NM, LI]),
    ("my usual grocery order, whats it cost now", [LH], ["Factual_Retrieval"], [NM, LI]),
    ("the model i was comparing against, has anything better come out", [SW], ["Analysis_&_Summarization"], [NM, LI, HC]),
    ("is that restaurant we talked about still open", [LH], ["Factual_Retrieval"], [NM, LI]),
    ("whats the current price of the thing on my wishlist", [BF], ["Factual_Retrieval"], [NM, LI]),
    ("the job i applied to, are they still hiring", [BF], ["Factual_Retrieval"], [NM, LI]),

    # --- hard negatives: sound referential but are NOT memory -------------
    ("what is the difference between a list and a tuple", [SW], ["Factual_Retrieval"], []),
    ("explain the previous decade of ai research", [SW], ["Analysis_&_Summarization"], [HC]),
    ("summarize this: the quick brown fox jumped over the lazy dog repeatedly", [GR], ["Analysis_&_Summarization"], []),
    ("whats the last element of a python list called", [SW], ["Factual_Retrieval"], []),
    ("in the previous example you gave, why is x used", [SW], ["Factual_Retrieval"], []),
    ("what did shakespeare write before hamlet", [GR], ["Factual_Retrieval"], []),
    ("continue this sentence: the rain in spain falls", [CR], ["Generation"], []),
    ("my code is throwing a keyerror, heres the whole file: def main(): d = {}; return d['x']", [SW], ["Troubleshooting"], []),
]

rows = [make_row(t, top, i, c, note="needs_memory_cross_conversation")
        for t, top, i, c in ROWS]
n = add(rows)
print(f"added {n} rows (of {len(rows)} offered)\n")
report()
