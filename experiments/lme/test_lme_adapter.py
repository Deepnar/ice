from __future__ import annotations

import sys
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace


LME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LME_DIR))

import lme_run


def test_instance_layout_preserves_session_boundaries_and_sorts_time():
    instance = {
        "question_id": "q1",
        "haystack_session_ids": ["later", "earlier"],
        "haystack_dates": ["2023/02/02 10:00", "2023/01/01 09:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "later user"},
                {"role": "assistant", "content": "later assistant"},
            ],
            [
                {"role": "user", "content": "earlier user"},
                {"role": "assistant", "content": "earlier assistant"},
            ],
        ],
    }

    layout = lme_run.instance_layout(instance, "oracle")
    sessions = layout["sessions"]

    assert layout["adapter_version"] == lme_run.ADAPTER_VERSION
    assert [session["session_id"] for session in sessions] == ["earlier", "later"]
    assert len({session["conversation_id"] for session in sessions}) == 2
    assert layout["query_conversation_id"] not in {
        session["conversation_id"] for session in sessions
    }
    assert sessions[0]["pairs"] == [("earlier user", "earlier assistant")]

    # Same benchmark identity must reproduce the same database layout.
    repeated = lme_run.instance_layout(instance, "oracle")
    assert repeated["query_conversation_id"] == layout["query_conversation_id"]
    assert [s["conversation_id"] for s in repeated["sessions"]] == [
        s["conversation_id"] for s in sessions
    ]


def test_answer_path_uses_new_auto_conversation_shape(monkeypatch):
    captured = {}

    class FakeClassification:
        topic_tags = ["Business_&_Finance"]
        intent_tags = ["Factual_Retrieval"]
        context_reliance = "Long_Term_Memory"

    class FakeClassifier:
        def classify(self, question, conversation_id=None):
            captured["classify"] = (question, conversation_id)
            return FakeClassification()

    class FakeEmbedder:
        def encode(self, _text, convert_to_tensor=False):
            assert convert_to_tensor is False
            return SimpleNamespace(tolist=lambda: [0.1, 0.2])

    class FakeQuery:
        def filter_by(self, **_kwargs):
            return self

        def all(self):
            return []

    class FakeDB:
        def query(self, _model):
            return FakeQuery()

        def close(self):
            captured["db_closed"] = True

    class FakeOrchestrator:
        def __init__(self, _db, _embedder):
            self.max_retrieval_tokens = None
            self.recent_token_budget = None

        def set_budget_from_turn_count(self, turns, total_tokens,
                                       classification=None):
            captured["budget"] = (turns, total_tokens, classification)
            self.max_retrieval_tokens = 2000
            self.recent_token_budget = 4240

        def retrieve(self, **kwargs):
            captured["retrieve"] = kwargs
            return []

    def fake_assemble_prompt(**kwargs):
        captured["assemble"] = kwargs
        return [{"role": "user", "content": kwargs["user_message"]}]

    orchestrator_module = ModuleType("src.retrieval.orchestrator")
    orchestrator_module.HybridRetrievalOrchestrator = FakeOrchestrator
    assembler_module = ModuleType("src.api.prompt_assembler")
    assembler_module.assemble_prompt = fake_assemble_prompt
    monkeypatch.setitem(sys.modules, "src.retrieval.orchestrator", orchestrator_module)
    monkeypatch.setitem(sys.modules, "src.api.prompt_assembler", assembler_module)

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion)
        )
    )
    query_cid = uuid.uuid4()

    result = lme_run.answer_instance(
        {"question": "What did I earn?"},
        query_cid,
        "full_ice",
        FakeClassifier(),
        FakeEmbedder(),
        lambda: FakeDB(),
        client,
        {"MemorySlot": object()},
    )

    assert captured["classify"][1] == str(query_cid)
    assert isinstance(captured["classify"][1], str)
    assert captured["budget"][:2] == (0, 0)
    assert captured["retrieve"]["conversation_id"] == str(query_cid)
    assert captured["retrieve"]["scope"] == {}
    assert captured["assemble"]["conversation_id"] == query_cid
    assert captured["assemble"]["scope"] == {}
    assert result["retrieval_budget"] == 2000
    assert result["recent_budget"] == 4240
    assert captured["db_closed"] is True
