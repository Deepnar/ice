"""Dynamic Model Registry – Hugging Face API + local model tagging, with per‑model backend URL."""

import json, os, time
import httpx
from openai import OpenAI
from src.api.config import settings
from src.workers.bg_client_factory import get_bg_client

REGISTRY_PATH = "models/model_registry.json"

# Mapping from Hugging Face tags to ICE topic labels
HF_TOPIC_MAP = {
    "code": "Software_&_Tech",
    "coding": "Software_&_Tech",
    "programming": "Software_&_Tech",
    "python": "Software_&_Tech",
    "text-generation": "General_Reference_&_Trivia",
    "conversational": "General_Reference_&_Trivia",
    "creative": "Creative_&_Media",
    "roleplay": "Creative_&_Media",
    "storytelling": "Creative_&_Media",
    "science": "STEM_&_Academics",
    "math": "STEM_&_Academics",
    "finance": "Business_&_Finance",
    "medical": "Lifestyle_&_Health",
}

# Mapping from Hugging Face tags to ICE intent labels
HF_INTENT_MAP = {
    "text-generation": "Generation",
    "conversational": "Casual_Banter",
    "code": "Generation",
    "coding": "Generation",
    "programming": "Generation",
    "roleplay": "Generation",
    "storytelling": "Generation",
    "science": "Factual_Retrieval",
    "math": "Factual_Retrieval",
    "finance": "Analysis_&_Summarization",
    "medical": "Analysis_&_Summarization",
}

def _default_registry():
    return {"models": {}, "updated_at": None}

def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return _default_registry()
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def save_registry(reg):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)

def _fetch_hf_tags(model_id: str) -> dict | None:
    """Return topic/intent tags from Hugging Face model card, or None if unavailable."""
    try:
        resp = httpx.get(f"https://huggingface.co/api/models/{model_id}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            raw_tags = data.get("tags", [])
            if raw_tags:
                return _map_hf_tags_to_ice(raw_tags)
    except Exception:
        pass
    return None

def _map_hf_tags_to_ice(tags):
    topic_tags = set()
    intent_tags = set()
    for tag in tags:
        if tag in HF_TOPIC_MAP:
            topic_tags.add(HF_TOPIC_MAP[tag])
        if tag in HF_INTENT_MAP:
            intent_tags.add(HF_INTENT_MAP[tag])
    return {"topic_tags": list(topic_tags), "intent_tags": list(intent_tags)}

def populate_from_ollama():
    """Fetch installed models from Ollama, tag via HF or background model, and fill the registry."""
    reg = load_registry()
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        models = resp.json().get("models", [])
    except Exception:
        models = []

    bg_client = get_bg_client()
    for m in models:
        name = m.get("name", m.get("model", "unknown"))
        if name in reg["models"]:
            continue

        # 1) Try Hugging Face
        suggestion = _fetch_hf_tags(name)
        confirmed = False

        # 2) Fall back to background model
        if not suggestion or (not suggestion.get("topic_tags") and not suggestion.get("intent_tags")):
            try:
                completion = bg_client.chat.completions.create(
                    model="Qwen/Qwen2.5-3B-Instruct-AWQ",
                    messages=[
                        {"role": "system", "content": (
                            "You are a model classifier. Given a model name, suggest its best topic and intent tags. "
                            "Output ONLY a JSON object with keys 'topic_tags' (list) and 'intent_tags' (list)."
                        )},
                        {"role": "user", "content": name},
                    ],
                    temperature=0.0, max_tokens=150, timeout=15.0,
                )
                raw = completion.choices[0].message.content.strip()
                suggestion = json.loads(raw)
            except Exception:
                suggestion = {"topic_tags": [], "intent_tags": []}
        else:
            confirmed = True  # HF models are pre‑confirmed

        reg["models"][name] = {
            "topic_tags": suggestion.get("topic_tags", []),
            "intent_tags": suggestion.get("intent_tags", []),
            "priority": 5,
            "context_window": 8192,
            "confirmed": confirmed,
            "base_url": None,           # None means use the default Ollama URL
            "added_at": time.time(),
        }

    reg["updated_at"] = time.time()
    save_registry(reg)
    return reg

def find_best_model(topic_tags, intent_tags):
    """Return (model_name, base_url_or_None) for the best matching confirmed model."""
    reg = load_registry()
    best_model = None
    best_score = -1
    best_url = None
    for name, entry in reg["models"].items():
        if not entry.get("confirmed", False):
            continue
        topic_overlap = len(set(topic_tags) & set(entry.get("topic_tags", [])))
        intent_overlap = len(set(intent_tags) & set(entry.get("intent_tags", [])))
        score = topic_overlap + intent_overlap + entry.get("priority", 0)
        if score > best_score:
            best_score = score
            best_model = name
            best_url = entry.get("base_url")
    if best_model is None:
        best_model = get_fallback_model()
        best_url = None
    return best_model, best_url

def get_fallback_model():
    """Return the first confirmed model name, or the configured default."""
    reg = load_registry()
    for name, entry in reg["models"].items():
        if entry.get("confirmed", False):
            return name
    return settings.default_fallback_model