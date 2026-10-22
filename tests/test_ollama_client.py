"""Ollama Cloud client. Transport is fake; these tests must not open a socket."""

import json

import pytest

from class_quality.ollama_client import (
    JudgeResponse,
    OllamaCloudClient,
    OllamaConfigError,
    OllamaResponseError,
)

VALID_JUDGEMENT = {
    "criteria": [
        {
            "id": "opening",
            "rating": 2,
            "explanation": "Greeting at the start.",
            "timestamps": ["00:00:30"],
        }
    ]
}


class ScriptedTransport:
    def __init__(self, events):
        self.events = list(events)
        self.calls = []

    def post(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        if not self.events:
            raise AssertionError("unexpected extra request")
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


def _client(transport, **kwargs):
    return OllamaCloudClient(
        api_key=kwargs.pop("api_key", "test-key"),
        transport=transport,
        permitted=kwargs.pop("permitted", True),
        **kwargs,
    )


def _ok(content):
    body = json.dumps({"message": {"role": "assistant", "content": content}}).encode()
    return 200, body


def test_cloud_calls_are_refused_until_permission_is_explicit():
    transport = ScriptedTransport([_ok("{}")])
    client = _client(transport, permitted=False)

    with pytest.raises(PermissionError):
        client.complete_json("score this class")

    assert transport.calls == []


def test_missing_api_key_fails_before_a_request(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    transport = ScriptedTransport([_ok("{}")])
    client = OllamaCloudClient(api_key=None, transport=transport, permitted=True)

    with pytest.raises(OllamaConfigError, match="OLLAMA_API_KEY"):
        client.complete_json("score this class")

    assert transport.calls == []


def test_valid_json_is_validated_and_the_model_is_configurable():
    transport = ScriptedTransport([_ok(json.dumps(VALID_JUDGEMENT))])
    client = _client(transport, model="glm-5.3-flash", timeout=12)

    result = client.complete_json("score this class")

    assert isinstance(result, JudgeResponse)
    assert result.criteria[0].id == "opening"
    assert result.criteria[0].rating == 2
    call = transport.calls[0]
    assert call["url"] == "https://ollama.com/api/chat"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["payload"]["model"] == "glm-5.3-flash"
    assert call["payload"]["stream"] is False
    assert "format" not in call["payload"]
    assert call["timeout"] == 12


def test_default_model_is_gemma4_31b():
    transport = ScriptedTransport([_ok(json.dumps(VALID_JUDGEMENT))])
    client = _client(transport)

    client.complete_json("score this class")

    assert transport.calls[0]["payload"]["model"] == "gemma4:31b"


def test_fenced_json_is_accepted():
    fenced = "```json\n" + json.dumps(VALID_JUDGEMENT) + "\n```"
    transport = ScriptedTransport([_ok(fenced)])
    result = _client(transport).complete_json("score this class")
    assert result.criteria[0].rating == 2


def test_still_image_data_may_contain_slashes_but_recording_paths_may_not():
    image = "iVBOR" + ("w0KGgo/aaa" * 30)
    transport = ScriptedTransport([_ok(json.dumps(VALID_JUDGEMENT))])
    client = _client(transport)

    client.complete_json("describe the still image", images=[image])

    assert transport.calls[0]["payload"]["messages"][0]["images"] == [image]


def test_video_and_audio_paths_are_rejected_before_a_request():
    transport = ScriptedTransport([_ok("{}")])
    client = _client(transport)

    with pytest.raises(ValueError):
        client.complete_json("score this class", images=["C:/classes/session.mp4"])
    with pytest.raises(ValueError):
        client.complete_json("score this class", images=["session.wav"])

    assert transport.calls == []


def test_invalid_json_is_retried_once_and_then_reported():
    transport = ScriptedTransport([_ok("not json"), _ok("still not json")])

    with pytest.raises(OllamaResponseError):
        _client(transport).complete_json("score this class")

    assert len(transport.calls) == 2


def test_invalid_json_recovers_on_the_single_retry():
    transport = ScriptedTransport([_ok("not json"), _ok(json.dumps(VALID_JUDGEMENT))])
    result = _client(transport).complete_json("score this class")
    assert result.criteria[0].id == "opening"
    assert len(transport.calls) == 2


def test_a_criterion_with_only_an_id_stays_unscored_instead_of_failing_the_run():
    incomplete = json.dumps({"criteria": [{"id": "opening"}]})
    result = _client(ScriptedTransport([_ok(incomplete)])).complete_json("score this class")

    assert result.criteria[0].id == "opening"
    assert result.criteria[0].rating is None
    assert result.criteria[0].evidence == ""


def test_timeout_is_retried_once():
    transport = ScriptedTransport([TimeoutError("slow"), _ok(json.dumps(VALID_JUDGEMENT))])
    result = _client(transport).complete_json("score this class")
    assert result.criteria[0].rating == 2

    exhausted = ScriptedTransport([TimeoutError("slow"), TimeoutError("slow again")])
    with pytest.raises(OllamaResponseError):
        _client(exhausted).complete_json("score this class")
    assert len(exhausted.calls) == 2


def test_rate_limit_is_retried_once_and_not_more():
    transport = ScriptedTransport([(429, b"busy"), (429, b"still busy")])

    with pytest.raises(OllamaResponseError, match="429"):
        _client(transport).complete_json("score this class")

    assert len(transport.calls) == 2


def test_rate_limit_recovers_on_the_single_retry():
    transport = ScriptedTransport([(429, b"busy"), _ok(json.dumps(VALID_JUDGEMENT))])
    result = _client(transport).complete_json("score this class")
    assert result.criteria[0].rating == 2


def test_authentication_failure_is_not_retried():
    transport = ScriptedTransport([(401, b"nope"), _ok(json.dumps(VALID_JUDGEMENT))])

    with pytest.raises(OllamaResponseError, match="401"):
        _client(transport, api_key="super-secret-key").complete_json("score this class")

    assert len(transport.calls) == 1


def test_null_evidence_on_unscored_criteria_does_not_reject_the_response():
    payload = {
        "criteria": [
            {
                "id": "opening",
                "rating": 2,
                "explanation": "Greeting at the start.",
                "timestamps": ["00:00:30"],
                "evidence": "Hello class",
            },
            {
                "id": "tone",
                "rating": None,
                "explanation": "not assessed",
                "timestamps": None,
                "evidence": None,
            },
        ]
    }
    result = _client(ScriptedTransport([_ok(json.dumps(payload))])).complete_json("score this class")

    assert result.criteria[0].evidence == "Hello class"
    assert result.criteria[1].rating is None
    assert result.criteria[1].evidence == ""
    assert result.criteria[1].timestamps == []


def test_rules_context_reply_shapes_still_parse():
    shapes = [
        {
            "criteria": {
                "opening": {"rating": "2 / 2", "explanation": "Greeting.", "timestamps": "00:00:30", "evidence": "Hello"},
                "tone": {"rating": "not assessed", "explanation": None, "timestamps": None, "evidence": None},
            }
        },
        [{"id": "Opening and rapport", "rating": "2", "evidence": "Hello class"}],
        "Notes first.\n{\"rules\": {\"opening\": 1}}\n" + json.dumps(VALID_JUDGEMENT),
    ]
    for shape in shapes:
        content = shape if isinstance(shape, str) else json.dumps(shape)
        result = _client(ScriptedTransport([_ok(content)])).complete_json("score this class")
        assert result.criteria[0].id == "opening"
        assert result.criteria[0].rating == 2


def test_a_truncated_reply_keeps_the_complete_criteria():
    opening = {
        "id": "opening",
        "rating": 2,
        "explanation": "Greeting.",
        "timestamps": ["00:00:05"],
        "evidence": "Hello class",
    }
    truncated = '{"criteria": [' + json.dumps(opening) + ', {"id": "inclusion", "rating": 0, "explanation": "cut'
    result = _client(ScriptedTransport([_ok(truncated)])).complete_json("score this class")

    assert [(item.id, item.rating) for item in result.criteria] == [("opening", 2)]


def test_errors_do_not_echo_the_api_key():
    transport = ScriptedTransport([(500, b"super-secret-key")])
    client = _client(transport, api_key="super-secret-key")

    with pytest.raises(OllamaResponseError) as caught:
        client.complete_json("score this class")

    assert "super-secret-key" not in str(caught.value)
