"""Dynamic Model Registry – Hugging Face API + Ollama name mapping + background model tagging."""

import json, os, time, re
import httpx
import structlog
from src.api.config import settings

logger = structlog.get_logger("ice.model_registry")
from src.workers.bg_client_factory import (
    get_bg_client,
    get_bg_model_name,
    json_schema,
)

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


# ------------------------------------------------------------------
# Ollama → Hugging Face name mapping (best‑effort)
# ------------------------------------------------------------------
def _ollama_name_to_hf_id(name: str) -> str | None:
    """
    Try to convert an Ollama model tag to a Hugging Face repo ID.
    Returns the HF ID or None if no mapping is known.
    """
    # Remove Ollama tag suffix (everything after the first colon) and quantization suffix
    base = name.split(":")[0]

    # Known mappings (community + official)
    KNOWN = {
        "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",  # fallback if no size detected
        "qwen3": "Qwen/Qwen3-7B",
        "gemma4": "google/gemma-4-7b-it",
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "deepseek": "deepseek-ai/deepseek-llm-7b-chat",
    }

    # Try to extract size and variant from the tag (e.g., "32b-instruct-q4_K_M")
    size = None
    variant = ""
    if ":" in name:
        tag_part = name.split(":")[1]
        # Look for a size like "32b", "7b", "70b"
        size_match = re.search(r"(\d+)b", tag_part)
        if size_match:
            size = size_match.group(1) + "B"
        # Detect instruct / chat
        if "instruct" in tag_part:
            variant = "Instruct"
        elif "chat" in tag_part:
            variant = "Chat"

    if base in KNOWN:
        hf_id = KNOWN[base]
        # If we have a size, try to adjust the model name (very simplistic)
        if size:
            # Replace default size if present
            hf_id = re.sub(r"\d+B", size, hf_id)
        if variant:
            hf_id = hf_id.removesuffix("-Instruct").removesuffix("-Chat") + f"-{variant}"
        return hf_id

    # Heuristic for qwen* models: Qwen/Qwen<version>-<size>-Instruct
    if base.startswith("qwen"):
        version = base[len("qwen"):]  # e.g., "2.5", "3.6"
        if size:
            return f"Qwen/Qwen{version}-{size}-Instruct"
        return f"Qwen/Qwen{version}-7B-Instruct"

    return None


def _fetch_hf_tags(model_id: str) -> dict | None:
    """Return topic/intent tags from Hugging Face model card, or None if unavailable."""
    # Try with the original name
    resp = _call_hf_api(model_id)
    if resp:
        return resp

    # Try with the mapped HF ID
    hf_id = _ollama_name_to_hf_id(model_id)
    if hf_id:
        return _call_hf_api(hf_id)
    return None


def _call_hf_api(model_id: str) -> dict | None:
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


# ------------------------------------------------------------------
# Background model tagging (with detailed prompt)
# ------------------------------------------------------------------
def _tag_with_bg_model(name: str) -> dict:
    bg_client = get_bg_client()
    try:
        # This used to hardcode "Qwen/Qwen2.5-3B-Instruct-AWQ" — the *dedicated*
        # mode default — while get_bg_client() in the default *shared* mode
        # points at Ollama, which has no such model. Verified 2026-08-04: the
        # call returned `404 model not found` on every invocation and the bare
        # `except Exception` below turned that into empty tags, so background
        # model auto-tagging had silently never worked in the shipped config.
        completion = bg_client.chat.completions.create(
            model=get_bg_model_name(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a model classifier for a personal AI memory system. "
                        "Given a model name, you must suggest its best topic and intent tags. "
                        "Choose ONLY from the lists below. Output ONLY a JSON object with keys "
                        "'topic_tags' (list of 1‑3 strings) and 'intent_tags' (list of 1‑3 strings).\n\n"
                        "Topic tags:\n"
                        "- Software_&_Tech: code, debugging, dev tools, AI/ML\n"
                        "- STEM_&_Academics: math, science, research, studying\n"
                        "- Business_&_Finance: money, startups, work, management\n"
                        "- Creative_&_Media: writing, art, music, stories, games\n"
                        "- Admin_&_Productivity: scheduling, emails, organization, planning\n"
                        "- Lifestyle_&_Health: fitness, food, mental health, daily life\n"
                        "- Social_&_Relationships: people, emotions, communication, family\n"
                        "- World_&_Current_Events: news, geography, politics, history\n"
                        "- Meta_AI: asking about AI itself, how prompting works\n"
                        "- Null_Noise: gibberish, tests, empty messages\n"
                        "- General_Reference_&_Trivia: random facts, definitions, general knowledge\n\n"
                        "Intent tags:\n"
                        "- Factual_Retrieval: 'what is X', 'tell me about Y'\n"
                        "- Troubleshooting: 'this is broken, fix it', 'why doesn't this work'\n"
                        "- Generation: 'write me X', 'create Y', 'generate Z'\n"
                        "- Ideation: 'brainstorm ideas for', 'what are some ways to'\n"
                        "- Analysis_&_Summarization: 'summarize this', 'analyze that'\n"
                        "- Strategic_Planning: 'help me plan', 'what's the best approach for'\n"
                        "- Decision_Making: 'should I do X or Y', 'which is better'\n"
                        "- Emotional_Processing: venting, expressing feelings, seeking empathy\n"
                        "- Utility_Formatting: 'convert this to JSON', 'format this as a table'\n"
                        "- Casual_Banter: hello, thanks, jokes, small talk\n"
                        "- Open_Exploration: 'I wonder about X', 'let's think through Y together'\n\n"
                        "Example: for a creative writing model, output {\"topic_tags\":[\"Creative_&_Media\"],"
                        "\"intent_tags\":[\"Generation\",\"Ideation\"]}"
                    ),
                },
                {"role": "user", "content": name},
            ],
            temperature=0.0,
            max_tokens=150,
            timeout=15.0,
            response_format=json_schema("model_tags", {
                "type": "object",
                "properties": {
                    "topic_tags": {"type": "array", "items": {
                        "type": "string", "enum": sorted(set(HF_TOPIC_MAP.values()))}},
                    "intent_tags": {"type": "array", "items": {
                        "type": "string", "enum": sorted(set(HF_INTENT_MAP.values()))}},
                },
                "required": ["topic_tags", "intent_tags"],
            }),
        )
        raw = (completion.choices[0].message.content or "").strip()
        return json.loads(raw)
    except Exception as exc:
        # A fallback must be observable (CLAUDE.md): returning empty tags here
        # is indistinguishable from "this model genuinely has no tags", which
        # is how the 404 above stayed invisible.
        logger.warning("model_autotag_failed", model_name=name,
                       error=str(exc)[:200])
        return {"topic_tags": [], "intent_tags": []}


# ------------------------------------------------------------------
# Populate registry from Ollama
# ------------------------------------------------------------------
def populate_from_ollama():
    reg = load_registry()
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        models = resp.json().get("models", [])
    except Exception:
        models = []

    for m in models:
        name = m.get("name", m.get("model", "unknown"))
        if name in reg["models"]:
            continue

        # 1) Try Hugging Face (with Ollama→HF mapping)
        suggestion = _fetch_hf_tags(name)
        confirmed = False

        # 2) Fall back to background model
        if not suggestion or (not suggestion.get("topic_tags") and not suggestion.get("intent_tags")):
            suggestion = _tag_with_bg_model(name)
        else:
            confirmed = True  # HF models are pre‑confirmed

        reg["models"][name] = {
            "topic_tags": suggestion.get("topic_tags", []),
            "intent_tags": suggestion.get("intent_tags", []),
            "priority": 5,
            "context_window": 8192,
            "confirmed": confirmed,
            "base_url": None,
            "added_at": time.time(),
        }

    reg["updated_at"] = time.time()
    save_registry(reg)
    return reg


def find_best_model(topic_tags, intent_tags, required_tokens: int = 0):
    """Return (model_name, base_url_or_None) for the best matching confirmed model
    that can accommodate the required token count."""
    reg = load_registry()
    best_model = None
    best_score = -1
    best_url = None
    for name, entry in reg["models"].items():
        if not entry.get("confirmed", False):
            continue
        if required_tokens > 0 and entry.get("context_window", 0) < required_tokens:
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


def get_model_context_window(model_name: str):
    """Return the registry's context_window for *model_name*, or None if the
    model is unknown (C16: the budget derives from this instead of a constant)."""
    reg = load_registry()
    entry = reg["models"].get(model_name)
    if entry:
        return entry.get("context_window")
    return None


def get_fallback_model():
    """Return the first confirmed model name, or the configured default."""
    reg = load_registry()
    for name, entry in reg["models"].items():
        if entry.get("confirmed", False):
            return name
    return settings.default_fallback_model