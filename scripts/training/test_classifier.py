# scripts/training/test_classifier.py

import os
import sys

# Add src/ to the Python path so we can import classifier directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from classifier.classifier import PyTorchClassifier, ClassificationResult

HARD_TEST_PROMPTS = [
    # 1. Vague anaphora with emotional undertones
    "i still think about what u said, not sure if it helps or makes it worse",

    # 2. Code with personal reference ("my config") + LTM trigger
    "my nvim config from last week broke again after the arch update, same error as before",

    # 3. Sarcastic thanks (is it Casual_Banter or Emotional_Processing? context?)
    "oh great, another failed deploy, thanks vllm, ur the best",

    # 4. Extremely short – could be Null_Noise or LTM depending on context
    "k",

    # 5. Real_Time_Search with implicit “current” – no explicit temporal word
    "did the mercury finally dip below 40 here or am i still melting",

    # 6. Meta + emotional – asking about AI memory and feeling abandoned
    "do u even remember my name or am i just another session to u",

    # 7. Factual retrieval masked as casual curiosity about a personal topic
    "whats the difference between a lager and an ale again? my dad used to brew and i never asked",

    # 8. LTM where the only signal is “that” in a very short prompt
    "can we try that one more time",

    # 9. RTS with “this weekend” – borderline live data
    "any good concerts happening this weekend? i need to get out of the house",

    # 10. Null_Noise but with an actual word that could be Casual_Banter
    "hmm",

    # 11. Emotional_Processing with embedded LTM (past shared experience)
    "i just walked past that cafe where we had that awful date, i almost texted u",

    # 12. Technical Troubleshooting that could be Zero_Shot (self-contained) but has “my” (Signal B)
    "my pytorch training loop hangs at epoch 2 every single time, no error just stuck",

    # 13. Open_Exploration with Meta_AI crossover
    "what if we’re just training a tiny model inside a bigger simulation and ur the benchmark",

    # 14. Ambiguous “it” + live data request
    "is it still down or did they finally fix it",

    # 15. Extremely long, rambling personal prompt (tests max_len / truncation)
    "i woke up at 3am and couldnt stop thinking about the project and how we still havent fixed the auth flow and then i spiraled into wondering if i should even be doing this at all and maybe i should just go back to working at the café and forget about the whole startup thing, idk what to do anymore",

    # 16. Gibberish that looks like a typo but is actually random
    "asd;fkj",

    # 17. Fake‑out LTM – uses “we” but is a hypothetical
    "if we were to build a mars colony what would be the first 3 pieces of tech we need",

    # 18. LTM with clear Signal D (personal name) but Zero_Shot intent
    "is kael a good name for a fire mage or is it too generic",

    # 19. Real_Time_Search with domain‑specific tech update
    "did qwen3 drop yet or is it still the same old story",

    # 20. A prompt that is literally just a period
    ".",

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

classifier = PyTorchClassifier(model_path="models/classifier/ice_classifier_v3_qwen_ft3.pt")
for prompt in HARD_TEST_PROMPTS:
    result = classifier.classify(prompt)
    print(f"Prompt: {prompt}")
    print(f"Predicted topics: {result.topic_tags}")
    print(f"Predicted intents: {result.intent_tags}")
    print(f"Context reliance: {result.context_reliance}")
    print(f"Max confidence: {result.max_confidence:.4f}")
    print("-" * 80)

# Summary of context reliance
context_summary = {}
for prompt in HARD_TEST_PROMPTS:
    result = classifier.classify(prompt)
    context_reliance = result.context_reliance
    if context_reliance in context_summary:
        context_summary[context_reliance] += 1
    else:
        context_summary[context_reliance] = 1

print("Summary of context reliance:")
for context, count in context_summary.items():
    print(f"{context}: {count}")