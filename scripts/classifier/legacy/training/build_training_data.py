import json
import os

TOPIC_LABELS = [
    "Software_&_Tech", "STEM_&_Academics", "Business_&_Finance",
    "Creative_&_Media", "Admin_&_Productivity", "Lifestyle_&_Health",
    "Social_&_Relationships", "World_&_Current_Events", "Meta_AI",
    "Null_Noise", "General_Reference_&_Trivia"
]

INTENT_LABELS = [
    "Factual_Retrieval", "Troubleshooting", "Generation", "Ideation",
    "Analysis_&_Summarization", "Strategic_Planning", "Decision_Making",
    "Emotional_Processing", "Utility_Formatting", "Casual_Banter",
    "Open_Exploration"
]

CONTEXT_RELIANCE_LABELS = [
    "Zero_Shot", "Long_Term_Memory", "Real_Time_Search"
]


def convert_labels_to_vector(label_dict):
    """label_dict is the inner dict with keys: topic_labels, intent_labels, context_reliance."""
    topic_vec = [1.0 if lbl in label_dict.get("topic_labels", []) else 0.0 for lbl in TOPIC_LABELS]
    intent_vec = [1.0 if lbl in label_dict.get("intent_labels", []) else 0.0 for lbl in INTENT_LABELS]
    ctx = label_dict.get("context_reliance", "")
    ctx_vec = [1.0 if lbl == ctx else 0.0 for lbl in CONTEXT_RELIANCE_LABELS]
    return topic_vec + intent_vec + ctx_vec


def main():
    input_path = "data/labeled/labeled_prompts.jsonl"
    output_path = "data/labeled/training_data.jsonl"
    schema_path = "data/labeled/label_schema.json"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            label = item.get("label", {})
            topic_labels = label.get("topic_labels", [])
            intent_labels = label.get("intent_labels", [])

            # Skip orphan prompts with no topic or intent labels
            if not topic_labels or not intent_labels:
                continue

            label_vector = convert_labels_to_vector(label)

            out_obj = {
                "prompt": item["prompt"],
                "labels": label_vector,
                "source": item.get("source", "")
            }
            f_out.write(json.dumps(out_obj) + "\n")

    # Write label schema
    schema = {
        "topic_labels": TOPIC_LABELS,
        "intent_labels": INTENT_LABELS,
        "context_reliance_labels": CONTEXT_RELIANCE_LABELS
    }
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Training data saved to {output_path}")
    print(f"Label schema saved to {schema_path}")


if __name__ == "__main__":
    main()