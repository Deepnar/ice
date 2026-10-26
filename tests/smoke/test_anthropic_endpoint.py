"""FINAL's second judge: the cloud client speaks Anthropic Messages — system prompt in its
own field, text blocks joined, a cut reply raised as incomplete — and the haiku role
exists with the documented OpenCode Go model id."""

from types import SimpleNamespace as NS

import pytest

from experiments.lme.cloud_provider import CloudCompletionError, TextGenerator
from experiments.z2.profiles import Z2_PROFILES


class Fake:
    def __init__(self, stop="end_turn"):
        self.requests, self.stop = [], stop
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        return NS(id="m1", stop_reason=self.stop, usage=None,
                  content=[NS(type="text", text='{"grade": '), NS(type="text", text='"correct"}')])


def test_messages_request_and_reply():
    fake = Fake()
    gen = TextGenerator(Z2_PROFILES["haiku"], client=fake)
    out = gen.generate([dict(role="system", content="judge"), dict(role="user", content="q")],
                       temperature=0, max_output_tokens=100, session_id="s1")
    req = fake.requests[0]
    assert req["system"] == "judge" and req["messages"] == [dict(role="user", content="q")]
    assert req["model"] == "claude-haiku-5-5" and out.text == '{"grade": "correct"}'
    with pytest.raises(CloudCompletionError):
        TextGenerator(Z2_PROFILES["haiku"], client=Fake("max_tokens")).generate(
            [dict(role="user", content="q")], temperature=0, max_output_tokens=5, session_id="s1")
