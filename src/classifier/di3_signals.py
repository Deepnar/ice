"""DI3 Signal Extractors – lightweight, regex‑free density calculators."""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Sentiment word lists
# ---------------------------------------------------------------------------
SENTIMENT_WORDS = {
    "feel", "felt", "feeling", "frustrated", "upset", "angry", "happy",
    "sad", "love", "hate", "excited", "worried", "scared", "tired",
    "overwhelmed", "depressed", "anxious", "stressed", "hopeless",
    "grateful", "thankful", "annoyed", "irritated", "confused", "lost",
}

I_FEEL_PATTERNS = {"i feel", "i'm feeling"}
IM_SENTIMENT_PATTERNS = {"i'm", "im"}

# ---------------------------------------------------------------------------
# Code detection features
# ---------------------------------------------------------------------------
CODE_FEATURES = {
    "```": 0.4, "=": 0.1, "==": 0.1, "!=": 0.1, ">": 0.1, "<": 0.1,
    "def": 0.1, "class": 0.1, "function": 0.1, "import": 0.1,
    "{": 0.1, "}": 0.1, ";": 0.1,
    "print": 0.05, "return": 0.05, "if": 0.05, "else": 0.05,
    "for": 0.05, "while": 0.05,
}

# ---------------------------------------------------------------------------
# Meta‑AI features
# ---------------------------------------------------------------------------
META_KEYWORDS = {"you", "your", "model"}
META_PHRASES = {"prompt", "prompting"}
META_PATTERNS = {"how do i prompt", "what model", "which model", "how should i prompt"}

# ---------------------------------------------------------------------------
# Noise features
# ---------------------------------------------------------------------------
KEYBOARD_MASH = {"asdf", "qwerty", "asdfghjkl", "zzzzzzzz", "asd;fkj"}

# ---------------------------------------------------------------------------
# Reference (anaphora) words
# ---------------------------------------------------------------------------
REFERENCE_WORDS = {"this", "that", "it", "these", "those", "the"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_code_density(text: str) -> float:
    """Return 0.0 – 1.0 based on the presence of code tokens."""
    text_lower = text.lower()
    score = 0.0
    for token, weight in CODE_FEATURES.items():
        if token in text_lower:
            score += weight
    return min(score, 1.0)


def compute_sentiment_density(text: str) -> float:
    """Return 0.0 – 1.0 based on emotional language."""
    text_lower = text.lower()
    words = text_lower.split()
    score = 0.0

    # individual sentiment words
    for word in words:
        if word in SENTIMENT_WORDS:
            score += 0.1

    # "I feel" / "I'm feeling"
    for pattern in I_FEEL_PATTERNS:
        if pattern in text_lower:
            score += 0.2

    # "I'm" + sentiment word
    if any(p in text_lower for p in IM_SENTIMENT_PATTERNS):
        for word in words:
            if word in SENTIMENT_WORDS:
                score += 0.15

    return min(score, 1.0)


def compute_meta_density(text: str) -> float:
    """Return 0.0 – 1.0 for queries about the AI itself."""
    text_lower = text.lower()
    score = 0.0

    for kw in META_KEYWORDS:
        if kw in text_lower:
            score += 0.1
    for phrase in META_PHRASES:
        if phrase in text_lower:
            score += 0.15
    for pattern in META_PATTERNS:
        if pattern in text_lower:
            score += 0.2

    return min(score, 1.0)


def compute_noise_density(text: str) -> float:
    """Return 0.0 – 1.0 for gibberish / accidental inputs."""
    score = 0.0
    stripped = text.strip()

    if len(stripped) < 5:
        score += 0.2
    if not any(c.isalpha() for c in stripped):
        score += 0.6
    if any(mash in stripped.lower() for mash in KEYBOARD_MASH):
        score += 0.3
    if len(set(stripped)) <= 3 and len(stripped) > 3:
        score += 0.2

    return min(score, 1.0)


def compute_reference_density(text: str) -> float:
    """Return 0.0 – 1.0 for anaphoric / demonstrative references."""
    text_lower = text.lower()
    words = text_lower.split()
    score = 0.0
    word_scores = {"this": 0.15, "that": 0.1, "it": 0.05,
                   "these": 0.1, "those": 0.1, "the": 0.05}
    for word in words:
        if word in word_scores:
            score += word_scores[word]
    return min(score, 1.0)


def extract_signals(text: str) -> Dict[str, float]:
    """Compute all five density signals and return them in a dict."""
    return {
        "code_density": compute_code_density(text),
        "sentiment_density": compute_sentiment_density(text),
        "meta_density": compute_meta_density(text),
        "noise_density": compute_noise_density(text),
        "reference_density": compute_reference_density(text),
    }