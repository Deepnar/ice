from __future__ import annotations

import sys
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace


LME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LME_DIR))

import lme_run


def test_correct_profile_mute_is_retryable_not_a_profile_mismatch():
    profile = SimpleNamespace(
        model="gpt-5.6-luna",
        name="opencode-luna",
        endpoint="responses",
    )
    mute = {
        "status": "mute",
        "answer": "",
        "model": "gpt-5.6-luna",
        "provider_profile": "opencode-luna",
        "provider_endpoint": "responses",
    }

    assert lme_run.answer_identity_matches(mute, profile)
    assert not lme_run.answer_complete_for_profile(mute, profile)
    assert not lme_run.answer_identity_matches(
        {**mute, "model": "another-model"}, profile
    )


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

    class FakeGenerator:
        profile = SimpleNamespace(
            model="answer-model",
            name="answer-profile",
            endpoint="responses",
        )

        def generate(self, _messages, **_kwargs):
            captured["generation_kwargs"] = _kwargs
            return SimpleNamespace(
                text="answer",
                response_id="response-id",
                usage={"output_tokens": 7},
            )

    query_cid = uuid.uuid4()

    result = lme_run.answer_instance(
        {"question": "What did I earn?"},
        query_cid,
        "full_ice",
        FakeClassifier(),
        FakeEmbedder(),
        lambda: FakeDB(),
        FakeGenerator(),
        {"MemorySlot": object()},
        provider_session_id="stable-answer-session",
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
    assert result["model"] == "answer-model"
    assert result["provider_profile"] == "answer-profile"
    assert result["provider_endpoint"] == "responses"
    assert result["provider_usage"] == {"output_tokens": 7}
    assert result["provider_session_id"] == "stable-answer-session"
    assert captured["generation_kwargs"]["session_id"] == "stable-answer-session"
    assert captured["db_closed"] is True


def test_ollama_background_residency_evicts_other_models_and_pins_qwen(monkeypatch):
    loaded = iter([["gemma4:26b-a4b-it-q4_K_M"], ["qwen3:4b-instruct-bg"]])
    evicted = []
    monkeypatch.setattr(lme_run, "_ollama_loaded_models", lambda: next(loaded))
    monkeypatch.setattr(lme_run, "ollama_unload", evicted.append)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"response":"ok"}'

    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    lme_run.ensure_ollama_background_resident()

    assert evicted == ["gemma4:26b-a4b-it-q4_K_M"]
    assert captured["timeout"] == 120
    assert captured["request"].full_url.endswith("/api/generate")
    assert b"qwen3:4b-instruct-bg" in captured["request"].data
    assert b'"keep_alive": -1' in captured["request"].data
