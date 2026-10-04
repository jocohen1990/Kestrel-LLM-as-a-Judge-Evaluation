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

    def create(self, **request):
        self.calls.append(request)
        if self.reject_temperature and "extra_body" in request:
            raise FakeAPIError("temperature is not supported for this model", self.error_status)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")],
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
    assert "extra_body" not in client.calls[1]

    llm.generate("sys", "again", model="new-model")
    assert len(client.calls) == 3                      # remembered: no wasted call
    assert "extra_body" not in client.calls[2]


def test_other_errors_are_not_swallowed(monkeypatch):
    client = FakeClient(reject_temperature=True, error_status=500)
    use_client(monkeypatch, client)
    with pytest.raises(FakeAPIError):
        llm.generate("sys", "hi", model="model-b")
