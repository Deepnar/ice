from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


LME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LME_DIR))

import cloud_provider


def test_responses_adapter_extracts_output_and_never_exposes_key(monkeypatch):
    profile = cloud_provider.ProviderProfile(
        name="test-responses",
        endpoint="responses",
        model="model-r",
        base_url_env="TEST_BASE",
        api_key_env="TEST_KEY",
    )
    monkeypatch.setenv("TEST_BASE", "https://example.test/v1/")
    monkeypatch.setenv("TEST_KEY", "secret-value")
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp-1",
                output_text="  answer from responses  ",
                usage=SimpleNamespace(model_dump=lambda: {"output_tokens": 17}),
            )

    client = SimpleNamespace(responses=FakeResponses())
    result = cloud_provider.TextGenerator(profile, client=client).generate(
        [{"role": "user", "content": "question"}],
        temperature=0.2,
        max_output_tokens=123,
    )

    assert result.text == "answer from responses"
    assert result.response_id == "resp-1"
    assert result.usage == {"output_tokens": 17}
    assert captured["input"] == [{"role": "user", "content": "question"}]
    assert captured["max_output_tokens"] == 123
    assert "secret-value" not in repr(profile.metadata())
    assert profile.metadata()["base_url"] == "https://example.test/v1"


def test_chat_adapter_preserves_legacy_request_shape():
    profile = cloud_provider.ProviderProfile(
        name="test-chat",
        endpoint="chat_completions",
        model="model-c",
        base_url="http://localhost:9999/v1",
    )
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="chat-1",
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=" legacy answer ")
                )],
                usage=None,
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    result = cloud_provider.TextGenerator(profile, client=client).generate(
        [{"role": "system", "content": "system"},
         {"role": "user", "content": "question"}],
        temperature=0.7,
        max_output_tokens=456,
    )

    assert result.text == "legacy answer"
    assert captured["max_tokens"] == 456
    assert "input" not in captured


def test_omen_alpha_is_explicit_chat_completions_profile():
    profile = cloud_provider.get_profile("opencode-omen-alpha")
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="omen-1",
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=" OK ")
                )],
                usage=SimpleNamespace(model_dump=lambda: {"completion_tokens": 33}),
            )

    result = cloud_provider.TextGenerator(
        profile,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    ).generate(
        [{"role": "user", "content": "probe"}],
        temperature=0,
        max_output_tokens=16,
        session_id="omen-stable-session",
    )

    assert profile.model == "omen-alpha"
    assert profile.endpoint == "chat_completions"
    assert result.text == "OK"
    assert captured["model"] == "omen-alpha"
    assert captured["max_tokens"] == 16
    assert captured["extra_headers"] == {
        "x-opencode-session": "omen-stable-session"
    }


def test_opencode_profile_refuses_missing_session_header():
    profile = cloud_provider.get_profile("opencode-omen-alpha")
    generator = cloud_provider.TextGenerator(
        profile,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None))
        ),
    )
    with pytest.raises(ValueError, match="x-opencode-session"):
        generator.generate(
            [{"role": "user", "content": "probe"}],
            temperature=0,
            max_output_tokens=16,
        )


def test_responses_adapter_accepts_mapping_shaped_output():
    profile = cloud_provider.ProviderProfile(
        name="mapping-responses",
        endpoint="responses",
        model="model-r",
        base_url="https://example.test/v1",
    )

    class FakeResponses:
        def create(self, **_kwargs):
            return {
                "id": "resp-map",
                "output": [{
                    "content": [{"type": "output_text", "text": "mapped answer"}],
                }],
                "usage": {"output_tokens": 4},
            }

    result = cloud_provider.TextGenerator(
        profile, client=SimpleNamespace(responses=FakeResponses())
    ).generate(
        [{"role": "user", "content": "question"}],
        temperature=0,
        max_output_tokens=32,
    )

    assert result.text == "mapped answer"
    assert result.response_id == "resp-map"
    assert result.usage == {"output_tokens": 4}


def test_profile_can_omit_unsupported_temperature():
    profile = cloud_provider.ProviderProfile(
        name="no-temperature",
        endpoint="responses",
        model="reasoning-model",
        base_url="https://example.test/v1",
        supports_temperature=False,
    )
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(id="r", output_text="yes", usage=None)

    generator = cloud_provider.TextGenerator(
        profile, client=SimpleNamespace(responses=FakeResponses())
    )
    generator.generate(
        [{"role": "user", "content": "question"}],
        temperature=0,
        max_output_tokens=32,
    )

    assert "temperature" not in captured


def test_selected_env_loads_only_provider_keys(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "PROBE_API_KEY=abc\n"
        "PROBE_API_BASE_URL=https://example.test/v1\n"
        "DATABASE_URL=must-not-load\n"
    )
    monkeypatch.delenv("PROBE_API_KEY", raising=False)
    monkeypatch.delenv("PROBE_API_BASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    cloud_provider.load_selected_env(env_path)

    assert cloud_provider.os.environ["PROBE_API_KEY"] == "abc"
    assert cloud_provider.os.environ["PROBE_API_BASE_URL"] == "https://example.test/v1"
    assert "DATABASE_URL" not in cloud_provider.os.environ
