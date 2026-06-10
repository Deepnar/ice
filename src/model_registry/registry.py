"""Dynamic Model Registry – JSON file based, populated from Ollama."""

import json, os, time
import httpx
from openai import OpenAI
from src.api.config import settings
from src.workers.bg_client_factory import get_bg_client

REGISTRY_PATH = "models/model_registry.json"

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

def populate_from_ollama():
    """Fetch installed models from Ollama and fill the registry."""
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
        # Ask background model to suggest topic/intent affinities
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
                temperature=0.0,
                max_tokens=150,
                timeout=15.0,
            )
            raw = completion.choices[0].message.content.strip()
            suggestion = json.loads(raw)
        except Exception:
            suggestion = {"topic_tags": [], "intent_tags": []}

        reg["models"][name] = {
            "topic_tags": suggestion.get("topic_tags", []),
            "intent_tags": suggestion.get("intent_tags", []),
            "priority": 5,
            "context_window": 8192,
            "confirmed": False,
            "added_at": time.time(),
        }

    reg["updated_at"] = time.time()
    save_registry(reg)
    return reg

def find_best_model(topic_tags, intent_tags):
    """Select the highest-priority model whose tags overlap the most."""
    reg = load_registry()
    best_model = settings.default_fallback_model
    best_score = -1
    for name, entry in reg["models"].items():
        if not entry.get("confirmed", False):
            continue
        topic_overlap = len(set(topic_tags) & set(entry.get("topic_tags", [])))
        intent_overlap = len(set(intent_tags) & set(entry.get("intent_tags", [])))
        score = topic_overlap + intent_overlap + entry.get("priority", 0)
        if score > best_score:
            best_score = score
            best_model = name
    return best_model