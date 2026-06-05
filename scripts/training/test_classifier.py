# scripts/training/test_classifier.py

import os
import sys

# Add src/ to the Python path so we can import classifier directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from classifier.classifier import PyTorchClassifier, ClassificationResult

TEST_PROMPTS = [
    # ===== Long_Term_Memory signals =====
    "that bug we fixed last night is back, the authentication module still crashes",
    "u said the goo blade would dissolve before touching him, can we keep that detail",
    "continue from where we left off, i want the next scene to start right after the hug",
    "what was the name of my friend who helped with the dipex project? i keep forgetting",
    "remember when we decided to use postgres instead of sqlite? i need to explain why",
    "my character Kael, the one with the fire ability, what should his weakness be",
    "the plan we made for the newsletter launch, can u remind me the first step",
    "we talked about adding a reflection worker, can u write the skeleton for it",
    "this is the same issue i had yesterday with docker, the container keeps exiting",
    "make the dialogue feel like what we discussed, more natural and less formal",
    # ===== Real_Time_Search signals =====
    "what’s the current price of bitcoin? i need to decide whether to sell",
    "is python 3.13 out yet or still in rc? about to set up a new venv",
    "who won the cricket match today between india and australia?",
    "what’s the weather forecast for tomorrow in mumbai, i have a flight",
    "did the supreme court issue any rulings this week on privacy?",
    "are there any new releases of vllm today? i thought they said june 5",
    "what’s the latest on that flooding in bangkok? is the airport open",
    # ===== Ambiguous / implicit references (should be LTM) =====
    "will it work this time or do i need to change the whole approach",
    "i think this is the right way to go, what do u think",
    "can u make it shorter but keep the same meaning",
    "what do you think about the ending we wrote last month",
    "i tried that thing u suggested, didn't help at all",
    # ===== Rare classes =====
    "asdfghjkl",                     # Null_Noise
    "zzzzzzzz",                      # Null_Noise
    "test",                          # Null_Noise (but be careful, might be mislabeled)
    "do you actually remember me or do you just pretend",   # Meta_AI + Factual_Retrieval
    "what’s the best way to prompt you to get really creative answers", # Meta_AI + Factual_Retrieval
    "i wonder if a sentient AI would experience time differently than us", # Open_Exploration
    "what would happen if we could store entire lifetimes in memory", # Open_Exploration
    "haha that joke was terrible lmaoo",                # Casual_Banter
    "thanks a lot for helping with that, ur a lifesaver", # Casual_Banter
    "yo morning",                                       # Casual_Banter
    # ===== Multi-label complex prompts =====
    "fix the login bug and then write a summary of the changes for the team",
    "i need a plan to learn rust in 2 weeks, and also i'm feeling overwhelmed by my current project",
    "should i use mongodb or postgres for this social app, and can u write the schema for whichever u recommend",
    # ===== Edge cases =====
    "write a haiku about a frog",     # Zero_Shot, Creative_&_Media, Generation
    "translate 'hello' to japanese", # Zero_Shot, General_Reference_&_Trivia, Utility_Formatting
    "what is the capital of france", # Zero_Shot, General_Reference_&_Trivia, Factual_Retrieval (trivia, but casual enough)
]

classifier = PyTorchClassifier(model_path="models/classifier/ice_classifier_v2_ft_2.pt")
for prompt in TEST_PROMPTS:
    result = classifier.classify(prompt)
    print(f"Prompt: {prompt}")
    print(f"Predicted topics: {result.topic_tags}")
    print(f"Predicted intents: {result.intent_tags}")
    print(f"Context reliance: {result.context_reliance}")
    print(f"Max confidence: {result.max_confidence:.4f}")
    print("-" * 80)

# Summary of context reliance
context_summary = {}
for prompt in TEST_PROMPTS:
    result = classifier.classify(prompt)
    context_reliance = result.context_reliance
    if context_reliance in context_summary:
        context_summary[context_reliance] += 1
    else:
        context_summary[context_reliance] = 1

print("Summary of context reliance:")
for context, count in context_summary.items():
    print(f"{context}: {count}")