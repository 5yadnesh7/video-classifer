"""Rubric judge. Ollama is mocked. No dataset text and no live HTTP calls."""

import json
import os
import re

import pytest

from class_quality.features import Features, Mention
from class_quality.judge import EvidencePack, FrameStill, RubricJudge, apply_judgement, build_judge_prompt, cloud_enabled
from class_quality.ollama_client import OllamaCloudClient, OllamaResponseError
from class_quality.rubric import MediaStats
from class_quality.vtt import Cue


def _features(**overrides: object) -> Features:
    values: dict[str, object] = dict(
        duration_s=600.0,
        cue_count=8,
        student_turn_share=0.55,
        student_time_share=0.5,
        distinct_students=3,
        teacher_question_count=4,
        teacher_words_per_minute=150.0,
        homework_mentions=[Mention(500.0, "00:08:20")],
        greeting_mentions=[Mention(30.0, "00:00:30")],
        student_turns_in_opening=2,
        closing_mentions=[Mention(560.0, "00:09:20")],
        non_latin_letter_ratio=0.0,
        transcript_reliable=True,
        reliability_reasons=[],
    )
    values.update(overrides)
    return Features(**values)  # type: ignore[arg-type]


def _media() -> MediaStats:
    return MediaStats(
        duration_s=600.0,
        audio_present=True,
        video_present=True,
        loudness_lufs=-18.0,
        loudness_range_lu=8.0,
        silence_s=12.0,
        longest_silence_s=4.0,
        mean_luminance=80.0,
        dark_frame_ratio=0.0,
        static_frame_ratio=0.1,
        frame_count=4,
    )


def _cues() -> list[Cue]:
    return [
        Cue(30.0, 32.0, "TEACHER", "Hello, how are you?"),
        Cue(50.0, 52.0, "STUDENT_1", "I am fine."),
        Cue(500.0, 502.0, "TEACHER", "Please do the homework tonight."),
        Cue(560.0, 562.0, "TEACHER", "See you next class."),
    ]


def _pack(**overrides: object) -> EvidencePack:
    values: dict[str, object] = dict(
        cues=_cues(),
        features=_features(),
        media=_media(),
        frames=[],
        forbidden_names=["Pat Smith", "Pat", "Smith"],
    )
    values.update(overrides)
    return EvidencePack(**values)  # type: ignore[arg-type]


def _item(criterion_id: str, rating: int, timestamp: str, evidence: str) -> dict:
    return {
        "id": criterion_id,
        "rating": rating,
        "explanation": evidence,
        "timestamps": [timestamp],
        "evidence": evidence,
    }


def _strong_payload() -> dict:
    return {
        "classification": "Poor",
        "criteria": [
            _item("opening", 2, "00:00:30", "TEACHER greets the class"),
            _item("inclusion", 2, "00:00:50", "STUDENT_1 answers"),
            _item("tone", 2, "00:00:30", "The teacher sounds warm and energetic"),
            _item("closure", 2, "00:09:20", "TEACHER closes the class"),
            _item("homework", 2, "00:08:20", "Please do the homework tonight."),
            _item("teacher_presence", 2, "00:00:30", "The teacher is on camera"),
            _item("branding", 2, "00:00:30", "An 8x8 background is visible"),
            _item("technical", 2, "00:00:30", "Integrated loudness is -18.0 LUFS"),
        ],
    }


class ScriptedTransport:
    def __init__(self, events):
        self.events = list(events)
        self.calls = []

    def post(self, url, headers, payload, timeout):
        self.calls.append(payload)
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


def _http(content: str, status: int = 200):
    body = json.dumps({"message": {"role": "assistant", "content": content}}).encode()
    return status, body


def test_prompt_receives_the_rules_result_as_context():
    from class_quality.rubric import score_rules

    rules = score_rules(_features(), _media())
    prompt, images = build_judge_prompt(_pack(), rules)

    assert "Rules opening:" in prompt
    assert "Rules tone: not assessed." in prompt
    assert "context only" in prompt
    assert "Do not copy a rules rating." in prompt
    assert "Keep a different rating only" not in prompt
    for line in prompt.splitlines():
        if line.startswith("Rules ") and "not assessed" not in line:
            assert "/ 2" not in line
    assert images == []
    assert "Pat" not in prompt
    assert ".mp4" not in prompt


def test_prompt_asks_for_criterion_scores_and_leaves_the_class_to_code():
    prompt, images = build_judge_prompt(_pack())

    for criterion_id in ("opening", "inclusion", "tone", "closure", "homework", "teacher_presence", "branding", "technical"):
        assert criterion_id in prompt
    assert '"rating"' in prompt
    assert '"evidence"' in prompt
    assert "do not assign an overall class" in prompt.lower()
    assert "not assessed" in prompt.lower()
    assert "transcript text" in prompt.lower()
    assert images == []
    assert "Pat" not in prompt
    assert "00:08:20" in prompt
    assert "0.55" in prompt or "55%" in prompt


def test_short_repeated_and_overlapping_names_do_not_leak_into_the_prompt():
    named = [
        Cue(30.0, 32.0, "Bo Li", "Hello bo, Bo met Anna Lee and Ann."),
        Cue(40.0, 42.0, "A", "Hi BO and A?"),
        Cue(50.0, 52.0, "Ann", "Thanks Li."),
    ]
    prompt, images = build_judge_prompt(
        _pack(cues=named, forbidden_names=["Bo Li", "Bo", "Li", "A", "Anna Lee", "Anna", "Ann"])
    )
    transcript = prompt.split("Transcript:", 1)[1]

    for secret in ("Bo", "bo", "BO", "Li", "Anna", "Ann"):
        assert re.search(rf"\b{secret}\b", transcript) is None
    assert re.search(r"(?m)\bA\b", transcript) is None
    assert "TEACHER" in prompt
    assert "STUDENT_" in prompt
    assert "a criteria" in prompt or "Score each" in prompt
    assert images == []
    assert ".mp4" not in prompt


def test_prompt_scrubs_names_and_omits_recording_paths():
    named = [
        Cue(30.0, 32.0, "Pat Smith", "Hello Pat Smith"),
        Cue(50.0, 52.0, "Quinn Jones", "I am fine?"),
    ]
    prompt, _ = build_judge_prompt(_pack(cues=named))

    for secret in ("Pat", "Smith", "Quinn", "Jones"):
        assert secret not in prompt
    assert "TEACHER" in prompt
    assert ".mp4" not in prompt


def test_missing_transcript_tells_the_judge_not_to_score_conversation():
    prompt, _ = build_judge_prompt(
        _pack(
            cues=[],
            features=_features(
                cue_count=0,
                student_turn_share=None,
                transcript_reliable=False,
                reliability_reasons=["No transcript was supplied."],
                greeting_mentions=[],
                closing_mentions=[],
                homework_mentions=[],
                distinct_students=0,
                student_turns_in_opening=0,
            ),
        )
    )
    assert "no transcript" in prompt.lower()


def test_model_class_is_ignored_and_scores_are_aggregated(monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    report = apply_judgement(_pack(), _strong_payload())

    assert report.mode == "hybrid"
    assert report.classification == "Excellent"
    assert report.score == pytest.approx(100.0)
    assert report.provisional is True
    tone = next(item for item in report.criteria if item.id == "tone")
    presence = next(item for item in report.criteria if item.id == "teacher_presence")
    branding = next(item for item in report.criteria if item.id == "branding")
    assert tone.status == "not_assessed"
    assert presence.status == "not_assessed"
    assert branding.status == "not_assessed"
    assert any("transcript text" in note.lower() for note in report.reliability_notes)
    assert "Pat" not in " ".join(item.explanation for item in report.criteria)


def test_invalid_score_missing_evidence_and_invented_timestamp_are_not_used():
    payload = {
        "criteria": [
            _item("opening", 3, "00:00:30", "TEACHER greets the class"),
            {**_item("inclusion", 2, "00:00:50", ""), "evidence": ""},
            _item("closure", 2, "00:44:44", "TEACHER closes the class"),
            _item("homework", 2, "00:00:30", "Please do the homework tonight."),
            _item("technical", 2, "00:00:30", "Integrated loudness is -99.0 LUFS"),
        ]
    }
    report = apply_judgement(_pack(), payload)
    by_id = {item.id: item for item in report.criteria}
    assert by_id["opening"].status == "not_assessed"
    assert by_id["inclusion"].status == "not_assessed"
    assert by_id["closure"].status == "assessed"
    assert by_id["closure"].rating == 2
    assert by_id["closure"].timestamps == []
    assert by_id["homework"].status == "not_assessed"
    assert by_id["technical"].status == "not_assessed"


def test_a_timestamp_inside_the_class_keeps_the_judge_rating():
    report = apply_judgement(_pack(), {"criteria": [_item("opening", 1, "00:03:00", "The teacher greets the class")]})
    opening = next(item for item in report.criteria if item.id == "opening")

    assert opening.status == "assessed"
    assert opening.rating == 1
    assert opening.timestamps == ["00:03:00"]


def test_invented_quotation_is_rejected_and_a_real_quote_is_kept():
    invented = _item("opening", 2, "00:00:30", 'TEACHER said "we flew to the moon today"')
    real = _item("opening", 1, "00:00:30", 'TEACHER said "Hello, how are you?"')
    assert apply_judgement(_pack(), {"criteria": [invented]}).criteria[0].status == "not_assessed"
    kept = apply_judgement(_pack(), {"criteria": [real]})
    opening = next(item for item in kept.criteria if item.id == "opening")
    assert opening.status == "assessed"
    assert opening.rating == 1
    assert opening.timestamps == ["00:00:30"]


def test_images_allow_visual_scores_only_at_known_frame_times():
    frame = FrameStill(timestamp_s=300.0, image_base64="iVBOR" + ("w0KGgo/aaa" * 5))
    pack = _pack(frames=[frame])
    prompt, images = build_judge_prompt(pack)
    assert images == [frame.image_base64]
    assert "00:05:00" in prompt
    assert ".mp4" not in prompt

    payload = {
        "criteria": [
            _item("branding", 2, "00:05:00", "The still shows a branded background"),
            _item("teacher_presence", 1, "00:05:00", "A person is in the sampled frame"),
            _item("branding", 2, "00:06:00", "Another frame shows the logo"),
        ]
    }
    report = apply_judgement(pack, payload)
    branding = [item for item in report.criteria if item.id == "branding"]
    assert len(branding) == 1
    assert branding[0].status == "assessed"
    assert branding[0].rating == 2
    presence = next(item for item in report.criteria if item.id == "teacher_presence")
    assert presence.status == "assessed"
    assert presence.rating == 1


def test_technical_measurement_without_a_timestamp_is_kept_when_loudness_matches():
    payload = {
        "criteria": [
            {
                "id": "technical",
                "rating": 2,
                "explanation": "Loudness is usable.",
                "timestamps": [],
                "evidence": "Integrated loudness is -18.0 LUFS",
            }
        ]
    }
    report = apply_judgement(_pack(), payload)
    technical = next(item for item in report.criteria if item.id == "technical")
    assert technical.status == "assessed"
    assert technical.rating == 2


def test_technical_zero_still_caps_the_class_without_a_model_class():
    payload = _strong_payload()
    payload["classification"] = "Excellent"
    payload["criteria"] = [
        item if item["id"] != "technical" else _item("technical", 0, "00:00:30", "Integrated loudness is -18.0 LUFS and the audio drops out")
        for item in payload["criteria"]
    ]
    report = apply_judgement(_pack(), payload)
    assert report.score > 80
    assert report.classification == "Average"
    assert report.cap_applied == "technical"


def test_unreliable_transcript_blocks_conversational_scores():
    pack = _pack(
        features=_features(
            transcript_reliable=False,
            reliability_reasons=["Transcript language does not look like English."],
        )
    )
    report = apply_judgement(pack, _strong_payload())
    for criterion_id in ("opening", "inclusion", "closure", "homework", "tone"):
        item = next(criterion for criterion in report.criteria if criterion.id == criterion_id)
        assert item.status == "not_assessed"
    assert report.classification is None


def test_cloud_permission_is_off_unless_explicitly_configured(monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    assert cloud_enabled() is False
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "true")
    assert cloud_enabled() is False
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    assert cloud_enabled() is True
    assert os.environ["OLLAMA_CLOUD_PERMITTED"] == "1"


def test_judge_does_not_call_the_client_without_permission(monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    transport = ScriptedTransport([_http(json.dumps(_strong_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    with pytest.raises(PermissionError):
        RubricJudge(client).score(_pack())

    assert transport.calls == []
    assert client.permitted is True


def test_judge_does_not_turn_permission_on_by_itself(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    transport = ScriptedTransport([_http(json.dumps(_strong_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=False)

    with pytest.raises(PermissionError):
        RubricJudge(client).score(_pack())

    assert transport.calls == []
    assert client.permitted is False


def test_mocked_success_retries_once_and_keeps_the_evidence_field(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    good = json.dumps(_strong_payload())
    transport = ScriptedTransport([_http("not json"), _http(good)])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True, model="gemma4:31b")

    report = RubricJudge(client).score(_pack())

    assert len(transport.calls) == 2
    assert transport.calls[0]["model"] == "gemma4:31b"
    assert report.classification == "Excellent"
    assert "images" not in transport.calls[0]["messages"][0]


def test_null_message_object_is_retried_once_and_can_recover(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    empty = (200, json.dumps({"message": None}).encode())
    transport = ScriptedTransport([empty, _http(json.dumps(_strong_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    report = RubricJudge(client).score(_pack())

    assert len(transport.calls) == 2
    assert report.classification == "Excellent"


def test_null_message_content_is_retried_once_and_can_recover(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    empty = (200, json.dumps({"message": {"role": "assistant", "content": None}}).encode())
    transport = ScriptedTransport([empty, _http(json.dumps(_strong_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    report = RubricJudge(client).score(_pack())

    assert len(transport.calls) == 2
    assert report.classification == "Excellent"


def test_fenced_json_without_a_newline_is_still_accepted(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    fenced = "```json" + json.dumps(_strong_payload()) + "```"
    transport = ScriptedTransport([_http(fenced)])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    report = RubricJudge(client).score(_pack())

    assert len(transport.calls) == 1
    assert report.classification == "Excellent"


def test_urllib_connect_timeout_is_retried_once(monkeypatch):
    import urllib.error
    import urllib.request

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(timeout)
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    client = OllamaCloudClient(api_key="test-key", permitted=True, timeout=5)

    with pytest.raises(OllamaResponseError, match="timed out"):
        RubricJudge(client).score(_pack())

    assert len(calls) == 2


def test_non_object_cloud_body_is_retried_once_and_can_recover(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    transport = ScriptedTransport([(200, b"null"), _http(json.dumps(_strong_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    report = RubricJudge(client).score(_pack())

    assert len(transport.calls) == 2
    assert report.classification == "Excellent"


def test_null_rating_marks_only_that_criterion_not_assessed(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    payload = _strong_payload()
    for item in payload["criteria"]:
        if item["id"] == "opening":
            item["rating"] = None
    body = _http(json.dumps(payload))
    transport = ScriptedTransport([body, body])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    report = RubricJudge(client).score(_pack())

    opening = next(item for item in report.criteria if item.id == "opening")
    inclusion = next(item for item in report.criteria if item.id == "inclusion")
    assert len(transport.calls) == 1
    assert opening.status == "not_assessed"
    assert opening.rating is None
    assert inclusion.status == "assessed"
    assert inclusion.rating == 2


def test_homework_survives_when_another_cue_shares_the_same_second():
    cues = _cues() + [Cue(500.4, 501.0, "STUDENT_1", "Okay, I will.")]
    report = apply_judgement(
        _pack(cues=cues),
        {"criteria": [_item("homework", 2, "00:08:20", "Please do the homework tonight.")]},
    )
    homework = next(item for item in report.criteria if item.id == "homework")
    assert homework.status == "assessed"
    assert homework.rating == 2
    assert homework.timestamps == ["00:08:20"]


def test_timeout_authentication_and_rate_limit_stop_after_the_client_retry(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")

    timed_out = ScriptedTransport([TimeoutError("slow"), TimeoutError("still slow")])
    client = OllamaCloudClient(api_key="test-key", transport=timed_out, permitted=True)
    with pytest.raises(OllamaResponseError):
        RubricJudge(client).score(_pack())
    assert len(timed_out.calls) == 2

    denied = ScriptedTransport([(401, b"no"), _http(json.dumps(_strong_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=denied, permitted=True)
    with pytest.raises(OllamaResponseError, match="401"):
        RubricJudge(client).score(_pack())
    assert len(denied.calls) == 1

    limited = ScriptedTransport([(429, b"busy"), (429, b"busy")])
    client = OllamaCloudClient(api_key="test-key", transport=limited, permitted=True)
    with pytest.raises(OllamaResponseError, match="429"):
        RubricJudge(client).score(_pack())
    assert len(limited.calls) == 2
