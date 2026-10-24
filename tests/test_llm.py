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

    def __init__(self, reject_temperature=False, error_status=400, reject_forced_tool=False):
        self.calls = []
        self.reject_temperature = reject_temperature
        self.reject_forced_tool = reject_forced_tool
        self.error_status = error_status
        self.messages = self   # so client.messages.create(...) works
        self.content = [SimpleNamespace(type="text", text="hello")]
        self.stop_reason = "end_turn"

    def create(self, **request):
        self.calls.append(request)
        if self.reject_temperature and "temperature" in request.get("extra_body", {}):
            raise FakeAPIError("temperature is not supported for this model", self.error_status)
        choice = request.get("extra_body", {}).get("tool_choice", {})
        if self.reject_forced_tool and choice.get("type") in ("tool", "any"):
            # The real message from Run 5
            raise FakeAPIError('tool_choice: type "tool" and "any" are not supported '
                               'for this model.', self.error_status)
        return SimpleNamespace(
            content=self.content,
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=3, output_tokens=1),
        )


def use_client(monkeypatch, client):
    """Install a fake client and start with no remembered models."""
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm, "_NO_TEMPERATURE_MODELS", set())
    monkeypatch.setattr(llm, "_NO_FORCED_TOOL_MODELS", set())


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


# ---- Models that reject a forced tool_choice (found in Run 5) -------------

def test_falls_back_to_auto_tool_choice_and_remembers(monkeypatch):
    client = FakeClient(reject_forced_tool=True)
    tool_reply(client)
    use_client(monkeypatch, client)
    result = llm.generate_structured("sys", "hi", SCHEMA, "submit", "d", model="judge")
    assert result.data == {"ok": True}
    assert len(client.calls) == 2
    retry = client.calls[1]["extra_body"]
    assert retry["tool_choice"] == {"type": "auto"}
    assert retry["tools"][0]["strict"] is True          # still strict
    assert llm.forces_tool_choice("judge") is False

    llm.generate_structured("sys", "again", SCHEMA, "submit", "d", model="judge")
    assert len(client.calls) == 3                        # remembered: no wasted call
    assert client.calls[2]["extra_body"]["tool_choice"] == {"type": "auto"}


def test_both_fallbacks_in_one_call(monkeypatch):
    """The Run 5 judge model rejects both temperature and a forced tool."""
    client = FakeClient(reject_temperature=True, reject_forced_tool=True)
    tool_reply(client)
    use_client(monkeypatch, client)
    llm.generate_structured("sys", "hi", SCHEMA, "submit", "d", model="judge")
    assert len(client.calls) == 3
    final = client.calls[2]["extra_body"]
    assert "temperature" not in final and final["tool_choice"] == {"type": "auto"}
    assert not llm.sends_temperature("judge") and not llm.forces_tool_choice("judge")


def test_auto_tool_choice_without_a_tool_call_is_an_error(monkeypatch):
    client = FakeClient(reject_forced_tool=True)   # replies with plain text
    use_client(monkeypatch, client)
    with pytest.raises(llm.StructuredOutputError):
        llm.generate_structured("sys", "hi", SCHEMA, "submit", "d", model="judge")


def test_tool_choice_errors_on_plain_calls_are_not_swallowed(monkeypatch):
    """No tool was forced, so there's nothing to relax: raise."""
    class Rejecting(FakeClient):
        def create(self, **request):
            self.calls.append(request)
            raise FakeAPIError("tool_choice: something else", 400)
    use_client(monkeypatch, Rejecting())
    with pytest.raises(FakeAPIError):
        llm.generate("sys", "hi", model="m")


def test_forced_tool_choice_reported_by_default(monkeypatch):
    use_client(monkeypatch, FakeClient())
    assert llm.forces_tool_choice("any-model") is True


# ---- Prompt caching (Run 8) -----------------------------------------------

def test_system_prompt_is_plain_text_unless_caching_is_asked_for(monkeypatch):
    client = FakeClient()
    use_client(monkeypatch, client)
    llm.generate("sys", "hi", model="m")
    assert client.calls[0]["system"] == "sys"


def test_cache_system_sends_a_cacheable_system_block(monkeypatch):
    client = FakeClient()
    tool_reply(client)
    use_client(monkeypatch, client)
    llm.generate_structured("sys", "hi", SCHEMA, "submit", "d", model="m", cache_system=True)
    assert client.calls[0]["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]


def test_input_tokens_include_cached_tokens(monkeypatch):
    """With caching, the API's input_tokens excludes cached tokens. Adding them
    back keeps 'Tokens in' comparable with runs made before caching."""
    client = FakeClient()
    use_client(monkeypatch, client)
    original = client.create

    def create(**request):
        r = original(**request)
        r.usage = SimpleNamespace(input_tokens=200, output_tokens=5,
                                  cache_read_input_tokens=600, cache_creation_input_tokens=0)
        return r
    client.create = create
    result = llm.generate("sys", "hi", model="m", cache_system=True)
    assert result.input_tokens == 800
    assert result.cache_read_input_tokens == 600 and result.cache_creation_input_tokens == 0


def test_usage_without_cache_fields_still_works(monkeypatch):
    use_client(monkeypatch, FakeClient())   # older SDKs / fakes have no cache fields
    result = llm.generate("sys", "hi", model="m")
    assert result.input_tokens == 3 and result.cache_read_input_tokens == 0


# ---- generate_with_tools (the Phase 4 agent) ---------------------------------

def test_generate_with_tools_returns_blocks_as_plain_dicts(monkeypatch):
    client = FakeClient()
    client.content = [
        SimpleNamespace(type="text", text="Let me check."),
        SimpleNamespace(type="tool_use", id="t1", name="calculator", input={"expression": "1+1"}),
    ]
    client.stop_reason = "tool_use"
    use_client(monkeypatch, client)
    messages = [{"role": "user", "content": "1+1?"}]
    tools = [{"name": "calculator", "description": "d", "input_schema": {"type": "object"}}]

    turn = llm.generate_with_tools("sys", messages, tools, model="model-a")

    assert turn.content == [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "1+1"}},
    ]
    assert turn.text == "Let me check." and len(turn.tool_calls) == 1
    assert turn.stop_reason == "tool_use"
    sent = client.calls[0]
    assert sent["messages"] == messages and sent["system"] == "sys"
    assert sent["extra_body"]["tools"] == tools
    assert "tool_choice" not in sent["extra_body"]   # the model decides: nothing forced


def test_generate_with_tools_still_falls_back_on_temperature(monkeypatch):
    client = FakeClient(reject_temperature=True)
    use_client(monkeypatch, client)
    turn = llm.generate_with_tools("sys", [{"role": "user", "content": "hi"}], [], model="model-b")
    assert turn.text == "hello" and turn.tool_calls == []
    assert "temperature" not in client.calls[-1]["extra_body"]
