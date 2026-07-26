#!/usr/bin/env python3
"""Pile B batch 4 — Meta_AI about ICE's own memory, plus remaining thin cells.

Meta_AI has 748 organic rows but they are all "what model are you", "how do i
prompt better" — website-chat meta. **Questions aimed at a system that actually
remembers have none**, because the corpus predates having one worth
interrogating. C10/C11 shipped the deletion + chat-command surface (`/forget`,
scope control, "what do you know about X"), and those turns are exactly what a
live ICE sees.

Also tops up the cells still thinnest after batches 1-3: Needs_Live_Info on its
own, Temporal_Recall WITHOUT a parseable date (the half the deterministic
detector is blind to, so only the label can catch it), and Admin/Lifestyle topics
that the corpus under-represents.
"""
import sys

sys.path.insert(0, "/home/deepnar/Programs/ice/scripts/classifier/pipeline")

from authored import add, make_row, report  # noqa: E402

SW, CR, BF = "Software_&_Tech", "Creative_&_Media", "Business_&_Finance"
LH, SR, ST = "Lifestyle_&_Health", "Social_&_Relationships", "STEM_&_Academics"
AP, MA, GR = "Admin_&_Productivity", "Meta_AI", "General_Reference_&_Trivia"
WC, NN = "World_&_Current_Events", "Null_Noise"
NM, TR, LI, HC = "Needs_Memory", "Temporal_Recall", "Needs_Live_Info", "High_Complexity"
FR, AN, UF, CB, OE, DM, EP, GEN = ("Factual_Retrieval", "Analysis_&_Summarization",
                                   "Utility_Formatting", "Casual_Banter",
                                   "Open_Exploration", "Decision_Making",
                                   "Emotional_Processing", "Generation")

ROWS = [
    # ── Meta_AI: interrogating what ICE remembers ─────────────────────────
    ("what do you know about me", [MA], [AN], [NM]),
    ("what do you remember about the ice project", [MA], [FR], [NM]),
    ("do you still have that thing i told you about my setup", [MA], [FR], [NM]),
    ("how much of our old conversations do you actually keep", [MA], [FR], []),
    ("whats stored about me right now", [MA], [FR], [NM]),
    ("did you save what i said about the deadline", [MA], [FR], [NM]),
    ("do you remember me", [MA], [FR], [NM]),
    ("list everything you have on the story project", [MA], [FR], [NM]),
    ("whats the oldest thing you remember about me", [MA], [FR], [NM, TR]),
    ("show me what you've been storing this whole time", [MA], [FR], [NM, TR]),
    ("are you keeping notes on our chats", [MA], [FR], []),
    ("what did you learn about me from that last conversation", [MA], [AN], [NM]),

    # ── Meta_AI: deletion / correction (C10/C11 surface) ──────────────────
    ("forget what i said about my job", [MA], [UF], [NM]),
    ("delete everything from that conversation", [MA], [UF], [NM]),
    ("stop remembering the stuff about my family", [MA], [UF], [NM]),
    ("that thing you saved about me is wrong, fix it", [MA], [UF], [NM]),
    ("wipe what you know about the old project", [MA], [UF], [NM]),
    ("dont store this one", [MA], [UF], []),
    ("can you make this chat private", [MA], [UF], []),
    ("i want to correct something you remember about my preferences", [MA], [UF], [NM]),
    ("remove the note about my salary", [MA], [UF], [NM]),
    ("actually forget that last bit", [MA], [UF], [NM]),

    # ── Meta_AI: capability / behaviour questions ─────────────────────────
    ("why did you bring that up, i didnt ask about it", [MA], [FR], [NM]),
    ("how did you know that about me", [MA], [FR], [NM]),
    ("are you searching my old chats for this", [MA], [FR], []),
    ("whats making you slow right now", [MA], [TS := "Troubleshooting"], []),
    ("why do you keep forgetting what i tell you", [MA], [EP], [NM]),
    ("can you actually remember across sessions or not", [MA], [FR], []),
    ("what model are you running", [MA], [FR], []),
    ("do you use the web or just what you know", [MA], [FR], []),

    # ── Temporal_Recall with NO parseable date (detector-blind half) ──────
    ("what was i thinking back when we started this", [SW], [FR], [NM, TR]),
    ("how did my plan change over time", [SW], [AN], [NM, TR]),
    ("what did i used to believe about this", [SW], [FR], [NM, TR]),
    ("has my opinion on this shifted at all", [SW], [AN], [NM, TR]),
    ("what was the original version of this idea", [CR], [FR], [NM, TR]),
    ("remind me how i felt about it in the beginning", [SR], [FR], [NM, TR]),
    ("whats different now compared to when we first talked", [SW], [AN], [NM, TR]),
    ("i keep going back and forth on this, whats the pattern", [SW], [AN], [NM, TR]),
    ("track how this project has drifted from the original scope", [SW], [AN], [NM, TR, HC]),
    ("when did i stop caring about that feature", [SW], [FR], [NM, TR]),
    ("early on i had a completely different approach, whats changed", [SW], [AN], [NM, TR]),
    ("whats something i was sure about that i was wrong on", [SW], [AN], [NM, TR, HC]),

    # ── Needs_Live_Info alone (no memory) ─────────────────────────────────
    ("whats the latest version of postgres", [SW], [FR], [LI]),
    ("is that library still being maintained", [SW], [FR], [LI]),
    ("whats bitcoin at right now", [BF], [FR], [LI]),
    ("did anything big happen in the news today", [WC], [FR], [LI]),
    ("hows the weather looking this weekend", [LH], [FR], [LI]),
    ("whats the current best open model for coding", [SW], [FR], [LI]),
    ("has anyone released anything interesting this week", [SW], [FR], [LI]),
    ("is the site down for everyone or just me", [SW], [TS], [LI]),
    ("whats the exchange rate today", [BF], [FR], [LI]),
    ("did the release ship yet", [SW], [FR], [LI]),
    ("whos winning the match", [LH], [FR], [LI]),
    ("is that conference sold out", [BF], [FR], [LI]),

    # ── Admin_&_Productivity (thin at 469) ────────────────────────────────
    ("help me plan out my week", [AP], [OE := "Strategic_Planning"], []),
    ("turn these notes into a proper agenda: standup, review, retro", [AP], [UF], []),
    ("what should i prioritise tomorrow given everything i told you", [AP], [DM], [NM]),
    ("draft a polite email declining the meeting", [AP], [GEN], []),
    ("reschedule things around the thing i mentioned", [AP], [UF], [NM]),
    ("make me a checklist for the launch", [AP], [GEN], [NM]),
    ("whats still on my list from before", [AP], [FR], [NM, TR]),
    ("summarise these three docs into one page", [AP], [AN], []),

    # ── Null_Noise (thin at 169, and it IS genuinely rare) ────────────────
    ("asdkjfhaskdjfh", [NN], [CB], []),
    ("...", [NN], [CB], []),
    ("test test", [NN], [CB], []),
    ("hello?????", [NN], [CB], []),
    ("ok", [NN], [CB], []),

    # ── Social / Lifestyle with memory (under-represented combo) ──────────
    ("hows things with the person i told you about", [SR], [FR], [NM]),
    ("did i ever tell you how that argument went", [SR], [FR], [NM, TR]),
    ("based on my goals whats a realistic training plan", [LH], [OE], [NM]),
    ("im stressed about the same thing again", [SR], [EP], [NM]),
    ("you know my dietary stuff, suggest something", [LH], [GEN], [NM]),
    ("whats my usual bedtime, am i slipping", [LH], [AN], [NM, TR]),

    # ── Creative with memory (under-represented vs its 7419 topic count) ──
    ("write the next scene in my usual voice", [CR], [GEN], [NM]),
    ("does this fit the tone of the earlier chapters", [CR], [AN], [NM]),
    ("what was that characters motivation again", [CR], [FR], [NM]),
    ("continue the story from where we stopped", [CR], [GEN], [NM]),
    ("is this consistent with the magic system i defined", [CR], [AN], [NM]),
]

rows = [make_row(t, top, i, c, note="meta_and_thin_cells") for t, top, i, c in ROWS]
n = add(rows)
print(f"added {n} rows (of {len(rows)} offered)\n")
report()
