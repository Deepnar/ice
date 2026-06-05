import json
from collections import Counter
import itertools

# Load labeled data
with open("/home/deepnar/Programs/ice/data/labeled/labeled_prompts.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Raw label frequencies
topic_labels = [label for item in data for label in item["label"]["topic_labels"]]
intent_labels = [label for item in data for label in item["label"]["intent_labels"]]
context_reliance = [item["label"]["context_reliance"] for item in data]
sources = [item["source"] for item in data]

topic_counts = Counter(topic_labels)
intent_counts = Counter(intent_labels)
context_counts = Counter(context_reliance)
source_counts = Counter(sources)

total_prompts = len(data)
topic_freq = {label: count / total_prompts for label, count in topic_counts.items()}
intent_freq = {label: count / total_prompts for label, count in intent_counts.items()}
context_freq = {label: count / total_prompts for label, count in context_counts.items()}
source_freq = {label: count / total_prompts for label, count in source_counts.items()}

print("Topic Frequencies:")
for label, freq in topic_freq.items():
    print(f"{label}: {freq:.2%}")

print("\nIntent Frequencies:")
for label, freq in intent_freq.items():
    print(f"{label}: {freq:.2%}")

print("\nContext Reliance Frequencies:")
for label, freq in context_freq.items():
    print(f"{label}: {freq:.2%}")

print("\nSource Frequencies:")
for label, freq in source_freq.items():
    print(f"{label}: {freq:.2%}")

# Co-occurrence matrices
topic_pairs = list(itertools.combinations(topic_labels, 2))
intent_pairs = list(itertools.combinations(intent_labels, 2))
topic_intent_pairs = [(topic, intent) for topic in topic_labels for intent in intent_labels]

topic_pair_counts = Counter(topic_pairs)
intent_pair_counts = Counter(intent_pairs)
topic_intent_pair_counts = Counter(topic_intent_pairs)

print("\nTop 20 Topic × Topic Co-occurrences:")
for pair, count in topic_pair_counts.most_common(20):
    print(f"{pair}: {count}")

print("\nTop 20 Intent × Intent Co-occurrences:")
for pair, count in intent_pair_counts.most_common(20):
    print(f"{pair}: {count}")

print("\nTop 20 Topic × Intent Co-occurrences:")
for pair, count in topic_intent_pair_counts.most_common(20):
    print(f"{pair}: {count}")

# Conditional distributions
context_by_source = {}
intent_by_topic = {}

for item in data:
    source = item["source"]
    context = item["label"]["context_reliance"]
    topic = item["label"]["topic_labels"][0] if item["label"]["topic_labels"] else None
    intent = item["label"]["intent_labels"][0] if item["label"]["intent_labels"] else None

    if source not in context_by_source:
        context_by_source[source] = Counter()
    context_by_source[source][context] += 1

    if topic and topic not in intent_by_topic:
        intent_by_topic[topic] = Counter()
    if topic and intent:
        intent_by_topic[topic][intent] += 1

print("\nContext Reliance by Source:")
for source, contexts in context_by_source.items():
    total = sum(contexts.values())
    print(f"Source: {source}")
    for context, count in contexts.items():
        print(f"  {context}: {count / total:.2%}")

print("\nIntent Distribution per Topic:")
for topic, intents in intent_by_topic.items():
    total = sum(intents.values())
    print(f"Topic: {topic}")
    for intent, count in intents.items():
        print(f"  {intent}: {count / total:.2%}")

# Label cardinality
avg_topic_labels = sum(len(item["label"]["topic_labels"]) for item in data) / total_prompts
avg_intent_labels = sum(len(item["label"]["intent_labels"]) for item in data) / total_prompts

print(f"\nAverage number of topic labels per prompt: {avg_topic_labels:.2f}")
print(f"Average number of intent labels per prompt: {avg_intent_labels:.2f}")

# Data completeness checks
no_topic_labels = sum(1 for item in data if not item["label"]["topic_labels"])
no_intent_labels = sum(1 for item in data if not item["label"]["intent_labels"])
invalid_context_reliance = sum(1 for item in data if item["label"]["context_reliance"] not in ["Zero_Shot", "Long_Term_Memory", "Real_Time_Search"])

print(f"\nNumber of prompts with no topic labels: {no_topic_labels}")
print(f"Number of prompts with no intent labels: {no_intent_labels}")
print(f"Number of prompts with invalid context reliance: {invalid_context_reliance}")

# Summary of skew
threshold = 0.02
infrequent_topics = [label for label, freq in topic_freq.items() if freq < threshold]
infrequent_intents = [label for label, freq in intent_freq.items() if freq < threshold]

print("\nLabels appearing less than 2% of the time:")
print("Topics:", infrequent_topics)
print("Intents:", infrequent_intents)

# Recommend synthetic data generation
print("\nRecommendations for synthetic data generation:")
if "Long_Term_Memory" not in topic_freq or topic_freq["Long_Term_Memory"] < threshold:
    print("- Increase the frequency of 'Long_Term_Memory' topics.")
if "Real_Time_Search" not in topic_freq or topic_freq["Real_Time_Search"] < threshold:
    print("- Increase the frequency of 'Real_Time_Search' topics.")
if "Emotional_Processing" not in intent_freq or intent_freq["Emotional_Processing"] < threshold:
    print("- Increase the frequency of 'Emotional_Processing' intents.")
if "Zero_Shot" not in context_freq or context_freq["Zero_Shot"] < threshold:
    print("- Increase the frequency of 'Zero_Shot' context reliance.")