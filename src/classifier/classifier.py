import torch
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List
from .model import ICEClassifier

@dataclass
class ClassificationResult:
    topic_tags: List[str]
    intent_tags: List[str]
    context_reliance: str
    raw_probs: List[float]        # 25 probabilities
    max_confidence: float
    prompt: str = ""

class PyTorchClassifier:
    def __init__(self, model_path="models/classifier/ice_classifier.pt",
                 schema_path="data/labeled/label_schema.json"):
        # These lists are fixed – the order must match training
        self.TOPIC_LABELS = [
            "Software_&_Tech", "STEM_&_Academics", "Business_&_Finance",
            "Creative_&_Media", "Admin_&_Productivity", "Lifestyle_&_Health",
            "Social_&_Relationships", "World_&_Current_Events", "Meta_AI",
            "Null_Noise", "General_Reference_&_Trivia"
        ]
        self.INTENT_LABELS = [
            "Factual_Retrieval", "Troubleshooting", "Generation", "Ideation",
            "Analysis_&_Summarization", "Strategic_Planning", "Decision_Making",
            "Emotional_Processing", "Utility_Formatting", "Casual_Banter",
            "Open_Exploration"
        ]
        self.CONTEXT_RELIANCE_LABELS = [
            "Zero_Shot", "Long_Term_Memory", "Real_Time_Search"
        ]

        # Load model on CPU
        self.model = ICEClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        # Embedder also on CPU
        # Embedder also on CPU – Qwen3-Embedding truncated to 384 dim for compatibility
        self.embedder = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device="cpu",
            truncate_dim=384
        )
    def classify(self, prompt: str, context_turns: list[str] | None = None) -> ClassificationResult:
        with torch.no_grad():
            # Build instruction‑prefixed prompt (Qwen3‑Embedding requires instruction format)
            prefixed = (
                "Classify the following user prompt into topic labels, intent labels, "
                "and determine whether it requires long-term memory, web search, or is self-contained:\n"
                f"{prompt}"
            )            
            # Encode and add batch dimension
            embedding = self.embedder.encode(prefixed, convert_to_tensor=True).unsqueeze(0).float()
            outputs = self.model(embedding)                     # (1, 25)

            topic_out = outputs[:, :11]                         # (1, 11)
            intent_out = outputs[:, 11:22]                      # (1, 11)
            ctx_out = outputs[:, 22:]                           # (1, 3)

            topic_probs = torch.sigmoid(topic_out).squeeze(0)   # (11,)
            intent_probs = torch.sigmoid(intent_out).squeeze(0) # (11,)
            ctx_probs = torch.softmax(ctx_out, dim=1).squeeze(0) # (3,)

        # Build tag lists
        topic_tags = [self.TOPIC_LABELS[i] for i in range(len(self.TOPIC_LABELS))
                      if topic_probs[i] > 0.3]
        intent_tags = [self.INTENT_LABELS[i] for i in range(len(self.INTENT_LABELS))
                       if intent_probs[i] > 0.3]
        if not topic_tags:
            topic_tags = [self.TOPIC_LABELS[torch.argmax(topic_probs).item()]]
        if not intent_tags:
            intent_tags = [self.INTENT_LABELS[torch.argmax(intent_probs).item()]]
        context_reliance = self.CONTEXT_RELIANCE_LABELS[torch.argmax(ctx_probs).item()]

        # Combine probabilities
        raw_probs = topic_probs.tolist() + intent_probs.tolist() + ctx_probs.tolist()
        max_confidence = max(raw_probs)
        # Safety net: story/lore questions always need long‑term memory
        if "Creative_&_Media" in topic_tags:
            context_reliance = "Long_Term_Memory"
            # Re‑compute max confidence from the raw probs (unchanged)
        if "Software_&_Tech" in topic_tags:
            referential_words = ["my", "our", "mine", "ours", "we", "us",
                "this", "that", "these", "those", "the",
                "it", "they", "them", "their",
                "previous", "last", "before", "yesterday", "earlier",
                "again", "still", "same"]
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in referential_words):
                context_reliance = "Long_Term_Memory"

        return ClassificationResult(
            topic_tags=topic_tags,
            intent_tags=intent_tags,
            context_reliance=context_reliance,
            raw_probs=raw_probs,
            max_confidence=max_confidence,
            prompt=prompt,
            
        )