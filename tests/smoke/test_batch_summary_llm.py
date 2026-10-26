"""G79: the batch summariser's model call takes the note writer's `schema` and sends it
as response_format, like conversation_summary._default_llm. Before the fix every batch
raised TypeError (caught) and no batch summary was ever written."""

from types import SimpleNamespace as NS

from src.workers import batch_summarizer as B


class FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return NS(choices=[NS(finish_reason="stop", message=NS(content='{"assertions": []}'))])


def test_batch_llm_accepts_schema_and_sends_response_format(monkeypatch):
    fake = FakeCompletions()
    monkeypatch.setattr(B, "bg_client", NS(chat=NS(completions=fake)))
    monkeypatch.setattr(B, "get_bg_model_name", lambda: "bg-model")
    monkeypatch.setattr(B.settings, "background_model_mode", "dedicated")
    monkeypatch.setattr(B.settings, "ollama_num_ctx_max", 32_768)
    schema = {"type": "object", "properties": {"assertions": {"type": "array"}}}
    B._batch_llm("Summarise these turns.", 512, schema=schema)
    assert fake.requests[-1]["response_format"]["type"] == "json_schema"
    B._batch_llm("Summarise these turns.", 512)
    assert "response_format" not in fake.requests[-1]
