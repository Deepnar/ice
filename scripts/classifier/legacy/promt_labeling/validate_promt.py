import json
from collections import Counter
from itertools import product

# ----------------------------------------------------------------------
# 1. LOAD DATA (skip empty lines)
# ----------------------------------------------------------------------
INPUT_PATH = "/home/deepnar/Programs/ice/data/labeled/labeled_prompts.jsonl"

with open(INPUT_PATH, "r") as f:
    data = [json.loads(line) for line in f if line.strip()]

total_prompts = len(data)
print(f"Total prompts loaded: {total_prompts}\n")

# ----------------------------------------------------------------------
# 2. BASIC FREQUENCIES (unchanged)
# ----------------------------------------------------------------------
def flatten_labels(items, key):
    """Extract list of labels from each item's label dict."""
    return [label for item in items for label in item["label"][key]]

topic_labels_flat = flatten_labels(data, "topic_labels")
intent_labels_flat = flatten_labels(data, "intent_labels")
context_vals = [item["label"]["context_reliance"] for item in data]
sources = [item["source"] for item in data]

topic_counts = Counter(topic_labels_flat)
intent_counts = Counter(intent_labels_flat)
context_counts = Counter(context_vals)
source_counts = Counter(sources)

def print_frequencies(counts, total, title):
    print(f"--- {title} ---")
    for label, count in counts.most_common():
        freq = count / total
        print(f"  {label:<35} {count:>6} ({freq:6.2%})")
    print()

print_frequencies(topic_counts, total_prompts, "Topic Frequencies")
print_frequencies(intent_counts, total_prompts, "Intent Frequencies")
print_frequencies(context_counts, total_prompts, "Context Reliance Frequencies")
print_frequencies(source_counts, total_prompts, "Source Frequencies")

# ----------------------------------------------------------------------
# 3. FULL CO‑OCCURRENCE MATRICES (topics, intents, topic‑intent)
# ----------------------------------------------------------------------
def build_cooccurrence_matrix(data, label_type1, label_type2=None):
    """
    Return a matrix (dict of dicts) of co‑occurrence counts.
    If label_type2 is None, it's a pairwise co‑occurrence within the same type
    (symmetrical).  Otherwise, asymmetric matrix where rows are label_type1
    and columns are label_type2.
    """
    all_labels1 = sorted(set(lbl for item in data for lbl in item["label"][label_type1]))
    if label_type2 is None:
        all_labels2 = all_labels1
    else:
        all_labels2 = sorted(set(lbl for item in data for lbl in item["label"][label_type2]))

    matrix = {l1: {l2: 0 for l2 in all_labels2} for l1 in all_labels1}

    for item in data:
        labs1 = item["label"][label_type1]
        if label_type2 is None:
            labs2 = labs1
            for i, l1 in enumerate(labs1):
                for j, l2 in enumerate(labs1):
                    if i < j:
                        matrix[l1][l2] += 1
                        matrix[l2][l1] += 1
        else:
            labs2 = item["label"][label_type2]
            for l1 in labs1:
                for l2 in labs2:
                    matrix[l1][l2] += 1
    return matrix

def print_cooccurrence_matrix(matrix, title):
    print(f"--- {title} ---")
    # column headers
    col_labels = sorted(list(matrix.values())[0].keys(), key=str)
    # print header row
    header = " " * 35 + "".join(f"{lbl[:20]:>22}" for lbl in col_labels)
    print(header)
    for row_label in sorted(matrix.keys(), key=str):
        row = matrix[row_label]
        vals = "".join(f"{row[col]:>22}" for col in col_labels)
        print(f"{row_label[:35]:<35}{vals}")
    print()

# Topic‑Topic co‑occurrence (symmetric)
tt_matrix = build_cooccurrence_matrix(data, "topic_labels")
print_cooccurrence_matrix(tt_matrix, "Topic × Topic Co‑occurrences")

# Intent‑Intent co‑occurrence (symmetric)
ii_matrix = build_cooccurrence_matrix(data, "intent_labels")
print_cooccurrence_matrix(ii_matrix, "Intent × Intent Co‑occurrences")

# Topic‑Intent co‑occurrence (asymmetric: rows = topic, cols = intent)
ti_matrix = build_cooccurrence_matrix(data, "topic_labels", "intent_labels")
print_cooccurrence_matrix(ti_matrix, "Topic (rows) × Intent (cols) Co‑occurrences")

# ----------------------------------------------------------------------
# 4. CONDITIONAL DISTRIBUTIONS (context by topic, context by intent)
# ----------------------------------------------------------------------
def context_by_label(data, label_type):
    """
    For each label, compute counts of each context_reliance value.
    Returns dict: label -> Counter of context values.
    """
    dist = {}
    for item in data:
        for lbl in item["label"][label_type]:
            dist.setdefault(lbl, Counter())[item["label"]["context_reliance"]] += 1
    return dist

print("--- Context Reliance by Topic ---")
ctx_by_topic = context_by_label(data, "topic_labels")
for topic in sorted(ctx_by_topic.keys()):
    total = sum(ctx_by_topic[topic].values())
    print(f"  {topic}:")
    for ctx, cnt in ctx_by_topic[topic].most_common():
        print(f"    {ctx}: {cnt} ({cnt/total:6.2%})")

print("\n--- Context Reliance by Intent ---")
ctx_by_intent = context_by_label(data, "intent_labels")
for intent in sorted(ctx_by_intent.keys()):
    total = sum(ctx_by_intent[intent].values())
    print(f"  {intent}:")
    for ctx, cnt in ctx_by_intent[intent].most_common():
        print(f"    {ctx}: {cnt} ({cnt/total:6.2%})")

# Existing context by source (slightly improved printing)
print("\n--- Context Reliance by Source ---")
ctx_by_src = {}
for item in data:
    src = item["source"]
    ctx_by_src.setdefault(src, Counter())[item["label"]["context_reliance"]] += 1
for src in sorted(ctx_by_src.keys()):
    total = sum(ctx_by_src[src].values())
    print(f"  {src}:")
    for ctx, cnt in ctx_by_src[src].most_common():
        print(f"    {ctx}: {cnt} ({cnt/total:6.2%})")

# ----------------------------------------------------------------------
# 5. CARDINALITY & COMPLETENESS
# ----------------------------------------------------------------------
avg_topic = sum(len(item["label"]["topic_labels"]) for item in data) / total_prompts
avg_intent = sum(len(item["label"]["intent_labels"]) for item in data) / total_prompts
print(f"\nAverage topic labels per prompt: {avg_topic:.2f}")
print(f"Average intent labels per prompt: {avg_intent:.2f}")

no_topic = sum(1 for item in data if not item["label"]["topic_labels"])
no_intent = sum(1 for item in data if not item["label"]["intent_labels"])
invalid_ctx = sum(1 for item in data if item["label"]["context_reliance"] not in
                  ["Zero_Shot", "Long_Term_Memory", "Real_Time_Search"])
print(f"Prompts with no topic labels: {no_topic}")
print(f"Prompts with no intent labels: {no_intent}")
print(f"Prompts with invalid context reliance: {invalid_ctx}")

# ----------------------------------------------------------------------
# 6. SKEW SUMMARY & RECOMMENDATIONS
# ----------------------------------------------------------------------
threshold = 0.02
infrequent_topics = [lbl for lbl, cnt in topic_counts.items() if cnt/total_prompts < threshold]
infrequent_intents = [lbl for lbl, cnt in intent_counts.items() if cnt/total_prompts < threshold]

print("\nLabels appearing in less than 2% of prompts:")
print(f"  Topics: {infrequent_topics if infrequent_topics else 'None'}")
print(f"  Intents: {infrequent_intents if infrequent_intents else 'None'}")

# Context reliance skew
ctx_freq = {ctx: cnt/total_prompts for ctx, cnt in context_counts.items()}
print("\nContext reliance balance check:")
if ctx_freq.get("Long_Term_Memory", 0) < 0.15:
    print("  ⚠️ Long_Term_Memory is below 15% — consider adding more LTM examples.")
if ctx_freq.get("Real_Time_Search", 0) < 0.01:
    print("  ⚠️ Real_Time_Search is below 1% — extremely underrepresented.")
else:
    print("  ✅ Context reliance distribution looks reasonable.")