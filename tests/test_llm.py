"""Test the temperature fallback in app/llm.py with a fake API client."""
from types import SimpleNamespace

import pytest

from app import llm


class FakeAPIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class FakeClient:
    """Records each request; rejects temperature if told to."""

    def __init__(self, reject_temperature=False, error_status=400):
        self.calls = []
        self.reject_temperature = reject_temperature
        self.error_status = error_status
        self.messages = self   # so client.messages.create(...) works
        self.content = [SimpleNamespace(type="text", text="hello")]
        self.stop_reason = "end_turn"

    def create(self, **request):
        self.calls.append(request)
        if self.reject_temperature and "temperature" in request.get("extra_body", {}):
            raise FakeAPIError("temperature is not supported for this model", self.error_status)
        return SimpleNamespace(
            content=self.content,
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=3, output_tokens=1),
        )


def use_client(monkeypatch, client):
    """Install a fake client and start with no remembered models."""
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm, "_NO_TEMPERATURE_MODELS", set())


def test_sends_temperature_via_extra_body(monkeypatch):
    client = FakeClient()
    use_client(monkeypatch, client)
    result = llm.generate("sys", "hi", model="model-a")
    assert result.text == "hello"
    assert client.calls[0]["extra_body"] == {"temperature": 0.0}


def test_retries_without_temperature_and_remembers(monkeypatch):
    client = FakeClient(reject_temperature=True)
    use_client(monkeypatch, client)

    assert llm.generate("sys", "hi", model="new-model").text == "hello"
    assert len(client.calls) == 2                      # failed try + retry
    assert "temperature" not in client.calls[1]["extra_body"]

    llm.generate("sys", "again", model="new-model")
    assert len(client.calls) == 3                      # remembered: no wasted call
    assert "temperature" not in client.calls[2]["extra_body"]


def test_other_errors_are_not_swallowed(monkeypatch):
    client = FakeClient(reject_temperature=True, error_status=500)
    use_client(monkeypatch, client)
    with pytest.raises(FakeAPIError):
        llm.generate("sys", "hi", model="model-b")


def test_sends_temperature_reports_models_that_rejected_it(monkeypatch):
    use_client(monkeypatch, FakeClient(reject_temperature=True))
    assert llm.sends_temperature("picky-model") is True    # not called yet
    llm.generate("sys", "hi", model="picky-model")
    assert llm.sends_temperature("picky-model") is False



# ---- Structured output (forced, strict tool call) -----------------------

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


def tool_reply(client, name="submit", data=None, stop_reason="tool_use"):
    client.content = [SimpleNamespace(type="tool_use", name=name, input=data or {"ok": True})]
    client.stop_reason = stop_reason


def test_structured_request_forces_one_strict_tool(monkeypatch):
    client = FakeClient()
    tool_reply(client)
    use_client(monkeypatch, client)
    result = llm.generate_structured("sys", "hi", SCHEMA, "submit", "Submit it", model="m")

    body = client.calls[0]["extra_body"]
    assert body["tools"] == [{"name": "submit", "description": "Submit it", "strict": True,
                              "input_schema": SCHEMA}]
    assert body["tool_choice"] == {"type": "tool", "name": "submit"}
    assert body["temperature"] == 0.0
    assert result.data == {"ok": True} and result.stop_reason == "tool_use"


def test_structured_keeps_the_tool_when_dropping_temperature(monkeypatch):
    client = FakeClient(reject_temperature=True)
    tool_reply(client)
    use_client(monkeypatch, client)
    llm.generate_structured("sys", "hi", SCHEMA, "submit", "d", model="picky")
    retry = client.calls[1]["extra_body"]
    assert "temperature" not in retry and retry["tool_choice"]["name"] == "submit"


def test_structured_raises_when_no_tool_call_comes_back(monkeypatch):
    client = FakeClient()   # plain text reply, e.g. cut off by max_tokens
    client.stop_reason = "max_tokens"
    use_client(monkeypatch, client)
    with pytest.raises(llm.StructuredOutputError, match="max_tokens"):
        llm.generate_structured("sys", "hi", SCHEMA, "submit", "d", model="m")


def test_structured_ignores_a_call_to_a_different_tool(monkeypatch):
    client = FakeClient()
    tool_reply(client, name="something_else")
    use_client(monkeypatch, client)
    with pytest.raises(llm.StructuredOutputError):
        llm.generate_structured("sys", "hi", SCHEMA, "submit", "d", model="m")
