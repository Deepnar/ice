import json, uuid
from src.api.db import SessionLocal
from src.classifier.classifier import PyTorchClassifier
from src.retrieval.orchestrator import HybridRetrievalOrchestrator

db = SessionLocal()
classifier = PyTorchClassifier(
    model_path="models/classifier/ice_classifier_v2_final.pt",
    schema_path="data/labeled/label_schema.json",
)
orch = HybridRetrievalOrchestrator(db, classifier.embedder)

# Use two completely different prompts – both scoped to the same conversation
prompt_A = "What did we decide about the laptop?"
prompt_B = "What is the capital of France?"   # should not match anything

for prompt in [prompt_A, prompt_B]:
    classification = classifier.classify(prompt)
    print("prompt stored in classification:", repr(classification.prompt))
    emb = classifier.embedder.encode(prompt, convert_to_tensor=False).tolist()
    fragments = orch.retrieve(
        classification=classification,
        conversation_id=str(uuid.uuid4()),
        prompt_embedding=emb,
        scope={"conversation_id": "3976f0b7-a244-40d4-9f26-d1752e02d128"}   # 1,038 turns   # replace with a real conversation ID
    )
    print(f"Prompt: {prompt}")
    for i, f in enumerate(fragments[:3], 1):
        print(f"  {i}. score={f.score:.4f}  source={f.source_type}  id={f.source_batch_id}")
        print(f"     {f.text[:80]}...")
    print()
db.close()