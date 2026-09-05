"""Pinned text-generation adapters for the LongMemEval harness.

The benchmark needs both OpenAI API families.  OpenCode Go currently serves
Muse Spark through ``/responses`` while the same model returns HTTP 500 through
``/chat/completions``.  Treating both as a generic "OpenAI-compatible" client
therefore loses experiment identity and, for Muse, loses the model entirely.

Profiles contain only public routing metadata.  Credentials are resolved from
environment-variable names at runtime and are never returned by ``metadata``.
When the harness runs from the frozen v2 worktree, the selected variables are
loaded from the main repository's ``.env`` because the worktree deliberately
contains only the v2 database configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAIN_ENV = Path(__file__).resolve().parents[2] / ".env"


class ProviderAccessError(RuntimeError):
    """Authentication/quota failure that requires operator action before retry."""

    def __init__(self, profile: str, kind: str, status_code: int | None,
                 message: str):
        self.profile = profile
        self.kind = kind
        self.status_code = status_code
        super().__init__(
            f"{profile}: provider {kind} failure"
            f"{f' (HTTP {status_code})' if status_code else ''}: {message}"
        )


def _access_error(profile: str, exc: Exception) -> ProviderAccessError | None:
    status = getattr(exc, "status_code", None)
    message = str(exc)
    lowered = message.lower()
    quota_markers = (
        "quota", "rate limit", "usage limit", "limit reached",
        "limit exceeded", "insufficient_quota", "billing", "credits",
        "subscription limit",
    )
    if status in (401, 403):
        return ProviderAccessError(profile, "authentication", status, message)
    if status in (402, 429) or any(marker in lowered for marker in quota_markers):
        return ProviderAccessError(profile, "quota/rate-limit", status, message)
    return None


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    endpoint: str
    model: str
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 600.0
    supports_temperature: bool = True

    def metadata(self) -> dict[str, Any]:
        """Public run identity.  Deliberately cannot contain a credential."""
        return {
            "profile": self.name,
            "endpoint": self.endpoint,
            "model": self.model,
            "base_url": self.resolved_base_url(),
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "supports_temperature": self.supports_temperature,
            "requires_session_header": self.api_key_env == "PROBE_API_KEY",
        }

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.base_url_env:
            value = os.environ.get(self.base_url_env, "").strip()
            if value:
                return value.rstrip("/")
            raise RuntimeError(
                f"{self.name}: required base URL environment variable "
                f"{self.base_url_env} is not set"
            )
        raise RuntimeError(f"{self.name}: no base URL configured")

    def resolved_api_key(self) -> str:
        if not self.api_key_env:
            return "dummy"
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise RuntimeError(
                f"{self.name}: required credential environment variable "
                f"{self.api_key_env} is not set"
            )
        return value


PROFILES: dict[str, ProviderProfile] = {
    "local-gemma26": ProviderProfile(
        name="local-gemma26",
        endpoint="chat_completions",
        model="gemma4:26b-a4b-it-q4_K_M",
        base_url="http://localhost:11434/v1",
    ),
    "opencode-muse13": ProviderProfile(
        name="opencode-muse13",
        endpoint="responses",
        model="muse-spark-1.3-contributor",
        base_url_env="PROBE_API_BASE_URL",
        api_key_env="PROBE_API_KEY",
    ),
    "opencode-omen-alpha": ProviderProfile(
        name="opencode-omen-alpha",
        endpoint="chat_completions",
        model="omen-alpha",
        base_url_env="PROBE_API_BASE_URL",
        api_key_env="PROBE_API_KEY",
    ),
    "opencode-luna": ProviderProfile(
        name="opencode-luna",
        endpoint="responses",
        model="gpt-5.6-luna",
        base_url_env="PROBE_API_BASE_URL",
        api_key_env="PROBE_API_KEY",
        supports_temperature=False,
    ),
    "opencode-mimo25": ProviderProfile(
        name="opencode-mimo25",
        endpoint="chat_completions",
        model="mimo-v2.5",
        base_url_env="PROBE_API_BASE_URL",
        api_key_env="PROBE_API_KEY",
    ),
    "opencode-longcat20": ProviderProfile(
        name="opencode-longcat20",
        endpoint="chat_completions",
        model="longcat-2.0",
        base_url_env="PROBE_API_BASE_URL",
        api_key_env="PROBE_API_KEY",
    ),
    "opencode-qwen38-flash": ProviderProfile(
        name="opencode-qwen38-flash",
        endpoint="chat_completions",
        model="qwen3.8-flash",
        base_url_env="PROBE_API_BASE_URL",
        api_key_env="PROBE_API_KEY",
    ),
    "opencode-deepseek-v4-flash": ProviderProfile(
        name="opencode-deepseek-v4-flash",
        endpoint="chat_completions",
        model="deepseek-v4-flash",
        base_url_env="PROBE_API_BASE_URL",
        api_key_env="PROBE_API_KEY",
    ),
    "local-gemma12-judge": ProviderProfile(
        name="local-gemma12-judge",
        endpoint="chat_completions",
        model="gemma4:12b",
        base_url="http://localhost:11434/v1",
    ),
}


def load_selected_env(path: Path = MAIN_ENV) -> None:
    """Load only provider variables, without letting arbitrary .env keys leak.

    This is intentionally smaller than ``load_dotenv``: the v2 Settings class
    rejects unknown keys, and the main repository contains settings introduced
    after v2.  Existing process values always win.
    """
    if not path.is_file():
        return
    allowed = {
        profile.base_url_env
        for profile in PROFILES.values()
        if profile.base_url_env
    } | {
        profile.api_key_env
        for profile in PROFILES.values()
        if profile.api_key_env
    }
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


def get_profile(name: str) -> ProviderProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown provider profile {name!r}; choose one of: {choices}") from exc


def _model_dump(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    return None


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read an SDK object or a plain mapping without exposing provider details."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _responses_text(response: Any) -> str:
    direct = _field(response, "output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    chunks: list[str] = []
    for item in _field(response, "output", []) or []:
        for content in _field(item, "content", []) or []:
            if _field(content, "type") == "output_text":
                text = _field(content, "text")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks).strip()


@dataclass(frozen=True)
class Generation:
    text: str
    response_id: str | None
    usage: dict[str, Any] | None


class TextGenerator:
    """One pinned model behind one explicit API family."""

    def __init__(self, profile: ProviderProfile, client: Any | None = None):
        self.profile = profile
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=profile.resolved_base_url(),
                api_key=profile.resolved_api_key(),
                timeout=profile.timeout_seconds,
            )
        self.client = client

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_output_tokens: int,
        session_id: str | None = None,
    ) -> Generation:
        extra_headers = None
        if self.profile.api_key_env == "PROBE_API_KEY":
            if not session_id or not session_id.strip():
                raise ValueError(
                    f"{self.profile.name}: x-opencode-session requires a stable "
                    "non-empty session id"
                )
            extra_headers = {"x-opencode-session": session_id.strip()}
        try:
            if self.profile.endpoint == "chat_completions":
                request = dict(
                    model=self.profile.model,
                    messages=messages,
                    max_tokens=max_output_tokens,
                )
                if self.profile.supports_temperature:
                    request["temperature"] = temperature
                if extra_headers:
                    request["extra_headers"] = extra_headers
                response = self.client.chat.completions.create(**request)
                text = (response.choices[0].message.content or "").strip()
            elif self.profile.endpoint == "responses":
                request = dict(
                    model=self.profile.model,
                    input=messages,
                    max_output_tokens=max_output_tokens,
                )
                if self.profile.supports_temperature:
                    request["temperature"] = temperature
                if extra_headers:
                    request["extra_headers"] = extra_headers
                response = self.client.responses.create(**request)
                text = _responses_text(response)
            else:
                raise ValueError(
                    f"{self.profile.name}: unsupported endpoint "
                    f"{self.profile.endpoint!r}"
                )
        except Exception as exc:
            access_error = _access_error(self.profile.name, exc)
            if access_error:
                raise access_error from exc
            raise
        return Generation(
            text=text,
            response_id=_field(response, "id"),
            usage=_model_dump(_field(response, "usage")),
        )
