"""ICE v2 learned-head input and decoding, without database or DI3 rules."""

import json
from pathlib import Path

import torch


def build_input(prompt: str, context_text: str | None = None) -> str:
    if context_text:
        return (
            f"Conversation context (summarized):\n{context_text}\n\n"
            "Given the above conversation and the user's latest prompt, predict:\n"
            "1. TOPIC: what is the subject (Software_&_Tech, Creative_&_Media, etc.)\n"
            "2. INTENT: what is the user trying to do (Factual_Retrieval, Troubleshooting, etc.)\n"
            "3. CONTEXT RELIANCE: does the user need memory (Zero_Shot, Long_Term_Memory, Real_Time_Search)\n\n"
            f"User prompt: {prompt}"
        )
    return (
        "Given a user prompt, predict:\n"
        "1. TOPIC: what is the subject (Software_&_Tech, Creative_&_Media, etc.)\n"
        "2. INTENT: what is the user trying to do (Factual_Retrieval, Troubleshooting, etc.)\n"
        "3. CONTEXT RELIANCE: does the user need memory (Zero_Shot, Long_Term_Memory, Real_Time_Search)\n\n"
        f"User prompt: {prompt}"
    )


def predict_head(model, embedder, prompt: str, context_text: str | None = None):
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    embedding = embedder.encode(build_input(prompt, context_text), convert_to_tensor=True)
    if embedding.shape != (384,):
        raise ValueError("ICE v2 requires exactly 384 embedding coordinates")
    with torch.no_grad():
        outputs = model(embedding.unsqueeze(0).float())
        topic = torch.sigmoid(outputs[0, :11])
        intent = torch.sigmoid(outputs[0, 11:22])
        context = torch.softmax(outputs[0, 22:], dim=0)
    def tags(probs, labels):
        return [labels[i] for i, p in enumerate(probs) if p > 0.3] or [labels[probs.argmax().item()]]
    probabilities = topic.tolist() + intent.tolist() + context.tolist()
    return {
        "topic_tags": tags(topic, config["topic_labels"]),
        "intent_tags": tags(intent, config["intent_labels"]),
        "context_reliance": config["context_labels"][context.argmax().item()],
        "raw_probs": probabilities,
        "max_confidence": max(probabilities),
    }
