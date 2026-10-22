"""Gradio upload flow. No live cloud calls and no dataset files."""

import json
import time
from pathlib import Path

import pytest

import class_quality.app as app_module
from class_quality.app import analyse_upload, build_interface, validate_upload
from class_quality.media import MediaError
from class_quality.ollama_client import OllamaCloudClient
from class_quality.rubric import MediaStats

VTT = """WEBVTT

00:00:30.000 --> 00:00:32.000
Pat Smith: Hello, how are you?

00:00:50.000 --> 00:00:52.000
Quinn Jones: I am fine.

00:01:10.000 --> 00:01:12.000
Rae Stone: I am here.

00:08:20.000 --> 00:08:22.000
Pat Smith: Please do the homework tonight.

00:09:20.000 --> 00:09:22.000
Pat Smith: See you next class.
"""

MEDIA = MediaStats(
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

SECRETS = ("Pat", "Smith", "Quinn", "Jones", "Rae", "Stone")


@pytest.fixture(autouse=True)
def cloud_disabled(monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)


class Recorder:
    def __init__(self):
        self.events: list[tuple[float, str | None]] = []

    def __call__(self, fraction, desc=None, **kwargs):
        self.events.append((fraction, desc))


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


def _item(criterion_id: str, rating: int, timestamp: str, evidence: str) -> dict:
    return {
        "id": criterion_id,
        "rating": rating,
        "explanation": evidence,
        "timestamps": [timestamp],
        "evidence": evidence,
    }


def _payload() -> dict:
    return {
        "classification": "Poor",
        "criteria": [
            _item("opening", 2, "00:00:30", "Pat Smith greets the class"),
            _item("inclusion", 2, "00:00:50", "A student answers"),
            _item("tone", 2, "00:00:30", "Pat Smith sounds warm"),
            _item("closure", 2, "00:09:20", "The teacher closes the class"),
            _item("homework", 2, "00:08:20", "Please do the homework tonight."),
            _item("teacher_presence", 2, "00:00:30", "Pat Smith is on camera"),
            _item("branding", 2, "00:00:30", "A branded background is visible"),
            _item("technical", 2, "00:00:30", "Integrated loudness is -18.0 LUFS"),
        ],
    }


def _files(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "Pat Smith.mp4"
    transcript = tmp_path / "Pat Smith.vtt"
    video.write_bytes(b"synthetic-mp4")
    transcript.write_text(VTT, encoding="utf-8")
    return video, transcript


def _measure(seen: list[str]):
    def measure(path, **kwargs):
        seen.append(path)
        return MEDIA

    return measure


def test_upload_validation_rejects_missing_and_non_mp4_files(tmp_path: Path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    missing = tmp_path / "missing.mp4"

    assert validate_upload(None) is not None
    assert "mp4" in validate_upload(None).lower()
    assert "mp4" in validate_upload(str(notes)).lower()
    assert validate_upload(str(missing)) is not None
    video, transcript = _files(tmp_path)
    assert validate_upload(str(video), str(notes)) is not None
    assert "vtt" in validate_upload(str(video), str(notes)).lower()
    assert validate_upload(str(video), str(transcript)) is None


def test_oversized_recording_is_rejected_before_it_is_copied(tmp_path: Path, monkeypatch):
    """Inject a tiny ceiling so a synthetic file stands in for a multi-gigabyte upload."""
    video, _transcript = _files(tmp_path)
    limit = video.stat().st_size - 1
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", limit, raising=False)
    created: list[Path] = []
    copied: list[str] = []

    def factory():
        path = tmp_path / "oversized-job"
        path.mkdir()
        created.append(path)
        return str(path)

    real_copy = app_module.shutil.copyfile

    def spy(src, dst, *args, **kwargs):
        copied.append(str(dst))
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(app_module.shutil, "copyfile", spy)

    result = analyse_upload(
        str(video),
        None,
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert copied == []
    assert created == []
    assert "not copied" in result.lower()
    assert f"{video.stat().st_size} bytes" in result
    assert f"{limit} bytes" in result
    assert "8 GiB" in result
    assert "90" in result
    assert "Classification:" not in result
    assert "Pat" not in result
    assert "Smith" not in result


def test_oversized_transcript_is_rejected_before_it_is_copied(tmp_path: Path, monkeypatch):
    video, transcript = _files(tmp_path)
    limit = video.stat().st_size
    assert transcript.stat().st_size > limit
    created: list[Path] = []
    copied: list[str] = []

    def factory():
        path = tmp_path / "oversized-vtt"
        path.mkdir()
        created.append(path)
        return str(path)

    real_copy = app_module.shutil.copyfile

    def spy(src, dst, *args, **kwargs):
        copied.append(str(dst))
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(app_module.shutil, "copyfile", spy)

    result = analyse_upload(
        str(video),
        str(transcript),
        max_bytes=limit,
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert copied == []
    assert created == []
    assert "transcript" in result.lower()
    assert "not copied" in result.lower()
    assert "Classification:" not in result
    for secret in SECRETS:
        assert secret not in result


def test_recording_at_the_injected_limit_is_still_analysed(tmp_path: Path, monkeypatch):
    video, _transcript = _files(tmp_path)
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", video.stat().st_size)
    created: list[Path] = []

    def factory():
        path = tmp_path / "at-limit"
        path.mkdir()
        created.append(path)
        return str(path)

    result = analyse_upload(
        str(video),
        None,
        measure=_measure([]),
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert "Classification:" in result
    assert "rules_only" in result
    assert created and not created[0].exists()


def test_empty_recording_is_rejected_without_creating_a_workspace(tmp_path: Path):
    video = tmp_path / "empty.mp4"
    video.write_bytes(b"")
    created: list[Path] = []

    def factory():
        path = tmp_path / "empty-job"
        path.mkdir()
        created.append(path)
        return str(path)

    result = analyse_upload(
        str(video),
        None,
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert created == []
    assert "empty" in result.lower()
    assert "Classification:" not in result


def test_empty_transcript_is_rejected_without_creating_a_workspace(tmp_path: Path):
    video, _transcript = _files(tmp_path)
    blank = tmp_path / "blank.vtt"
    blank.write_bytes(b"")
    created: list[Path] = []

    def factory():
        path = tmp_path / "empty-vtt"
        path.mkdir()
        created.append(path)
        return str(path)

    result = analyse_upload(
        str(video),
        str(blank),
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert created == []
    assert "empty" in result.lower()
    assert "transcript" in result.lower()
    assert "Classification:" not in result


def test_invalid_upload_returns_an_error_without_creating_a_workspace(tmp_path: Path):
    def factory():
        raise AssertionError("invalid uploads must not create a workspace")

    result = analyse_upload(str(tmp_path / "clip.mov"), None, workspace_factory=factory)

    assert "mp4" in result.lower()
    assert "Classification:" not in result


def test_rules_only_fallback_explains_that_the_judge_is_unavailable(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    video, transcript = _files(tmp_path)
    seen: list[str] = []
    transport = ScriptedTransport([_http(json.dumps(_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)
    progress = Recorder()
    created: list[Path] = []

    def factory():
        path = tmp_path / "job"
        path.mkdir()
        created.append(path)
        return str(path)

    result = analyse_upload(
        str(video),
        str(transcript),
        progress=progress,
        client=client,
        measure=_measure(seen),
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert transport.calls == []
    assert client.permitted is True
    assert "rules_only" in result
    assert "Classification:" in result
    assert "provisional" in result.lower()
    assert "unavailable" in result.lower()
    assert "OLLAMA_CLOUD_PERMITTED" in result
    assert progress.events
    assert progress.events[-1][0] == 1
    assert any(0.6 <= fraction < 1 for fraction, _desc in progress.events)
    assert seen and Path(seen[0]).name == "upload.mp4"
    assert "Pat" not in seen[0]
    for secret in SECRETS:
        assert secret not in result
    assert created and not created[0].exists()


def test_workspace_is_removed_when_analysis_fails(tmp_path: Path):
    video, transcript = _files(tmp_path)
    created: list[Path] = []

    def factory():
        path = tmp_path / "failed-job"
        path.mkdir()
        created.append(path)
        return str(path)

    def explode(path, **kwargs):
        raise RuntimeError("Pat Smith measurement failed")

    result = analyse_upload(
        str(video),
        str(transcript),
        measure=explode,
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert "failed" in result.lower()
    assert "Pat" not in result
    assert "Smith" not in result
    assert created and not created[0].exists()


def test_unreadable_transcript_returns_an_error_without_leaking_a_workspace(tmp_path: Path):
    video, transcript = _files(tmp_path)
    transcript.write_bytes(b"\xff\xfe")
    created: list[Path] = []

    def factory():
        path = tmp_path / "bad-encoding"
        path.mkdir()
        created.append(path)
        return str(path)

    result = analyse_upload(
        str(video),
        str(transcript),
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert "failed" in result.lower()
    assert "Classification:" not in result
    assert not created or not created[0].exists()


def test_judge_failure_reports_progress_and_removes_the_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    video, transcript = _files(tmp_path)
    created: list[Path] = []

    def factory():
        path = tmp_path / "judge-failed"
        path.mkdir()
        created.append(path)
        return str(path)

    transport = ScriptedTransport([RuntimeError("Pat Smith judge exploded")])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)
    progress = Recorder()

    result = analyse_upload(
        str(video),
        str(transcript),
        progress=progress,
        client=client,
        measure=_measure([]),
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert "failed" in result.lower()
    assert "Classification:" not in result
    for secret in SECRETS:
        assert secret not in result
        for _fraction, desc in progress.events:
            assert secret not in (desc or "")
    assert created and not created[0].exists()
    assert progress.events[-1][1] is not None
    assert "fail" in progress.events[-1][1].lower()


def test_hybrid_measurement_failure_is_explained_and_cleaned_up(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    video, transcript = _files(tmp_path)
    created: list[Path] = []

    def factory():
        path = tmp_path / "measure-failed"
        path.mkdir()
        created.append(path)
        return str(path)

    def fail_measure(path, **kwargs):
        raise MediaError("Pat Smith ffmpeg failed")

    transport = ScriptedTransport([_http(json.dumps(_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    result = analyse_upload(
        str(video),
        str(transcript),
        client=client,
        measure=fail_measure,
        workspace_factory=factory,
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert "could not measure" in result.lower()
    assert "hybrid" in result
    assert "Classification:" in result
    for secret in SECRETS:
        assert secret not in result
        assert secret not in json.dumps(transport.calls)
    assert created and not created[0].exists()


def test_permitted_cloud_uses_mocked_scores_and_hides_speaker_names(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    video, transcript = _files(tmp_path)
    transport = ScriptedTransport([_http(json.dumps(_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True, model="gemma4:31b")

    result = analyse_upload(
        str(video),
        str(transcript),
        client=client,
        measure=_measure([]),
        workspace_factory=lambda: str(tmp_path / "cloud-job"),
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert len(transport.calls) == 1
    assert transport.calls[0]["model"] == "gemma4:31b"
    assert "images" not in transport.calls[0]["messages"][0]
    sent = json.dumps(transport.calls[0])
    assert "hybrid" in result
    assert "Excellent" not in result
    assert "higher rating was not used" in result
    assert "Good" in result
    assert "00:08:20" in result
    assert "not assessed" in result.lower()
    assert "unavailable" not in result.lower()
    for secret in SECRETS:
        assert secret not in result
        assert secret not in sent
    assert not (tmp_path / "cloud-job").exists()


def test_cloud_permission_does_not_encode_stills_for_upload(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    video, transcript = _files(tmp_path)
    transport = ScriptedTransport([_http(json.dumps(_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    def forbid_stills(*_args, **_kwargs):
        raise AssertionError("unblurred stills must not be encoded")

    monkeypatch.setattr("class_quality.app.extract_stills", forbid_stills)
    analyse_upload(
        str(video),
        str(transcript),
        client=client,
        measure=_measure([]),
        workspace_factory=lambda: str(tmp_path / "no-stills"),
    )

    assert transport.calls
    assert "images" not in transport.calls[0]["messages"][0]


def test_lowercase_names_in_the_model_text_are_removed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    video, transcript = _files(tmp_path)
    payload = _payload()
    payload["criteria"][0]["explanation"] = "thanks pat"
    payload["criteria"][0]["evidence"] = "thanks pat"
    transport = ScriptedTransport([_http(json.dumps(payload))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    result = analyse_upload(
        str(video),
        str(transcript),
        client=client,
        measure=_measure([]),
        workspace_factory=lambda: str(tmp_path / "case-job"),
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert "pat" not in result.lower()


def test_short_names_are_removed_from_the_report():
    import re

    from class_quality.app import _scrub

    text = _scrub("Hello Bo and A. Hi BO, Li, and Anna Lee saw Ann.", ["Bo", "Li", "A", "Anna Lee", "Ann"])

    for secret in ("Bo", "BO", "Li", "Anna", "Ann"):
        assert re.search(rf"\b{secret}\b", text) is None
    assert re.search(r"\bA\b", text) is None
    assert "and" in text
    assert "saw" in text


def test_deployment_host_and_port_are_configurable(monkeypatch):
    monkeypatch.setenv("CLASS_QUALITY_HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "10000")
    captured: dict = {}

    def launch(self, *args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(app_module.gr.Blocks, "launch", launch)
    app_module.main()

    assert captured.get("server_name") == "0.0.0.0"
    assert captured.get("server_port") == 10000
    assert captured.get("share") is False


def test_launch_stays_on_localhost_without_a_public_share(monkeypatch):
    monkeypatch.setenv("GRADIO_SHARE", "true")
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.delenv("CLASS_QUALITY_HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    captured: dict = {}

    def launch(self, *args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(app_module.gr.Blocks, "launch", launch)
    app_module.main()

    assert captured.get("server_name") == "127.0.0.1"
    assert captured.get("share") is False
    assert captured.get("max_file_size") == 8 * 1024**3
    assert captured.get("footer_links") == []
    assert build_interface().fill_width is True


def test_interface_describes_the_permission_gate_without_launching():
    demo = build_interface()
    rendered = "\n".join(
        value
        for block in demo.blocks.values()
        if isinstance(value := getattr(block, "value", None), str)
    )

    assert "OLLAMA_CLOUD_PERMITTED" in rendered
    assert "mp4" in rendered.lower()
    assert "8 GiB" in rendered
    assert "90" in rendered
    assert "ClassQuality AI" in rendered
    assert "Poor" in rendered and "Average" in rendered and "Good" in rendered and "Excellent" in rendered
    assert "Extract evidence" in rendered
    assert "Understand what makes a class effective." in rendered
    assert "Evaluate" in rendered and "Report" in rendered
    assert "several minutes" in rendered.lower()
    assert "Privacy and limitations" in rendered
    assert any(getattr(block, "label", None) == "Use AI judge" for block in demo.blocks.values())
    assert "500 MB" not in rendered
    assert demo.blocks


def test_dashboard_shows_only_real_report_fields():
    from dataclasses import replace

    from class_quality.app import format_dashboard
    from class_quality.rubric import CriterionResult, Report

    report = Report(
        classification=None,
        score=None,
        provisional=True,
        mode="rules_only",
        cap_applied=None,
        reliability_notes=["Insufficient evidence to assign a class."],
        criteria=[
            CriterionResult("tone", "Tone and energy", 15, None, "not_assessed", "Tone was not assessed.", []),
            CriterionResult("technical", "Technical quality", 10, 2, "assessed", "Audio is present.", ["00:01:00"]),
        ],
        strengths=[],
        improvements=[],
        measurements={"duration_s": 120.0, "student_turn_share": None, "loudness_lufs": -18.0, "homework_mentions": 0},
    )

    view = format_dashboard(report)
    rendered = " ".join(view.values())
    assert "Provisional — Rules Only" in view["rating"]
    assert "Class not assigned" in view["rating"]
    assert "Key information" in view["rating"]
    assert "Important notes" in view["rating"]
    assert "Evaluation criteria" in view["rubric"]
    assert "Measured signals" in view["metrics"]
    assert "Areas for improvement" in view["findings"]
    assert "1 of 2" in view["rating"]
    assert "62.5" not in rendered and "72.5" not in rendered
    assert "Language detected" not in view["rating"]
    assert "70.0" not in view["rating"]
    withheld = format_dashboard(replace(report, score=100.0))
    assert "100.0" in withheld["rating"]
    assert "not a class grade" in withheld["rating"].lower()
    assert "Insufficient evidence" in view["rating"]
    assert "Not assessed" in view["rubric"]
    assert "00:01:00" in view["rubric"]
    assert "enough evidence" in view["findings"].lower()
    assert "Duration" in view["metrics"]
    assert "Student participation" not in view["metrics"]
    assert "Pat" not in " ".join(view.values())

    assisted = format_dashboard(replace(report, mode="hybrid", classification="Good", score=70.0, strengths=["Inclusion: students answered."]))
    assert "AI-Assisted Assessment" in assisted["rating"]
    assert "Good" in assisted["rating"]
    assert "70.0" in assisted["rating"]
    assert "students answered" in assisted["findings"]
    assert "Ollama Cloud" in assisted["rating"]
    assert "stays on this computer" not in assisted["rating"].lower()

    local = format_dashboard(replace(
        report,
        measurements={
            **report.measurements,
            "transcript_source": "local",
            "detected_language": "en",
            "language_probability": 0.91,
        },
    ))
    assert "Transcribed locally" in local["rating"]
    assert "English" in local["rating"]
    assert "0.91" in local["rating"]
    assert "was not sent to Ollama Cloud" in local["rating"]


def test_page_shows_rules_ai_and_the_rules_owned_final_result(tmp_path: Path, monkeypatch):
    from class_quality.app import present_analysis

    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    video, transcript = _files(tmp_path)
    transport = ScriptedTransport([_http(json.dumps(_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    error, show_empty, show_results, rating, rubric, findings, metrics = present_analysis(
        str(video),
        str(transcript),
        client=client,
        measure=_measure([]),
        workspace_factory=lambda: str(tmp_path / "both-views"),
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert error == ""
    assert show_empty is False and show_results is True
    assert "Rules result" in rating
    assert "AI result" in rating
    assert "Final result" not in rating
    opening = rubric.split("Inclusion")[0]
    assert "Rules: 1 / 2" in opening
    assert "higher judge rating was not used" in opening
    assert "AI: 2 / 2" not in opening
    assert "Greeting cues in the first 8 minutes: 1" in opening
    assert "Ollama Cloud" in rating
    assert "stays on this computer" not in rating.lower()
    rendered = " ".join((rating, rubric, findings, metrics))
    for secret in SECRETS:
        assert secret not in rendered
    assert not (tmp_path / "both-views").exists()


def test_ai_card_shows_the_judge_result_and_the_final_card_shows_the_combination():
    from class_quality.app import format_comparison
    from class_quality.rubric import CriterionResult, Report

    def item(criterion_id, name, weight, rating, status, explanation):
        return CriterionResult(criterion_id, name, weight, rating, status, explanation, [])

    shared = dict(provisional=True, cap_applied=None, strengths=[], improvements=[], measurements={"duration_s": 2274.0})
    rules = Report(
        classification="Poor",
        score=42.3,
        mode="rules_only",
        reliability_notes=["Rules-only mode is provisional. Tone, camera presence, and branding are not assessed."],
        criteria=[
            item("opening", "Opening and rapport", 15, 1, "assessed", "Rules opening."),
            item("inclusion", "Inclusion and individual engagement", 20, 0, "assessed", "Rules inclusion."),
            item("tone", "Tone and energy", 15, None, "not_assessed", "Tone and energy were not assessed in rules-only mode."),
            item("teacher_presence", "Teacher presence on camera", 15, None, "not_assessed", "No face detector was applied."),
            item("branding", "Professional setup", 5, None, "not_assessed", "Branded background was not assessed."),
        ],
        **shared,
    )
    ai = Report(
        classification="Excellent",
        score=86.4,
        mode="hybrid",
        reliability_notes=["Rapport and tone were not established from transcript text alone."],
        criteria=[
            item("opening", "Opening and rapport", 15, 1, "assessed", "AI opening."),
            item("inclusion", "Inclusion and individual engagement", 20, 2, "assessed", "AI inclusion."),
            item("tone", "Tone and energy", 15, None, "not_assessed", "Tone cannot be established from transcript text alone."),
            item("teacher_presence", "Teacher presence on camera", 15, None, "not_assessed", "No still images were judged."),
            item("branding", "Professional setup", 5, None, "not_assessed", "No still images were judged."),
        ],
        **shared,
    )

    view = format_comparison(rules, ai)
    ai_section = view["rating"].split("AI result", 1)[1]

    assert "Excellent" in ai_section
    assert "86.4" in ai_section
    assert "Final result" not in view["rating"]
    assert "rules result was given to the judge as context" in view["rating"].lower()
    assert "Rules only" not in view["rating"]
    assert "Rules-only mode is provisional" not in view["rating"]
    assert "not assessed not assessed" not in view["rubric"].lower()
    assert "how the teacher sounded" in view["rubric"]
    assert "faces can be blurred" in view["rubric"]


def test_ai_card_does_not_show_100_when_the_judge_skipped_a_low_rules_score():
    from class_quality.app import format_comparison
    from class_quality.rubric import CriterionResult, Report, align_ai_with_rules

    def item(criterion_id, name, weight, rating, status, explanation):
        return CriterionResult(criterion_id, name, weight, rating, status, explanation, [])

    shared = dict(provisional=True, cap_applied=None, strengths=[], improvements=[], measurements={"duration_s": 600.0, "frame_count": 3, "teacher_words_per_minute": 140.0})
    criteria = [
        ("opening", "Opening and rapport", 15, 2, "assessed", "Opening."),
        ("inclusion", "Inclusion and individual engagement", 20, 0, "assessed", "Students spoke 30%."),
        ("closure", "Lesson structure and closure", 10, 2, "assessed", "Closure."),
        ("homework", "Homework assigned", 10, 2, "assessed", "Homework."),
        ("technical", "Technical quality", 10, 2, "assessed", "Audio is present."),
        ("tone", "Tone and energy", 15, None, "not_assessed", "Tone was not assessed."),
        ("teacher_presence", "Teacher presence on camera", 15, None, "not_assessed", "No face check."),
        ("branding", "Professional setup", 5, None, "not_assessed", "No background check."),
    ]
    rules = Report(
        classification="Good",
        score=69.2,
        mode="rules_only",
        reliability_notes=["Rules-only mode is provisional. Tone, camera presence, and branding are not assessed."],
        criteria=[item(*values) for values in criteria],
        **shared,
    )
    ai_criteria = []
    for criterion_id, name, weight, rating, status, explanation in criteria:
        if criterion_id == "inclusion":
            ai_criteria.append(item(criterion_id, name, weight, None, "not_assessed", "No usable evidence."))
        else:
            ai_criteria.append(item(criterion_id, name, weight, rating, status, explanation))
    ai = Report(classification="Excellent", score=100.0, mode="hybrid", reliability_notes=[], criteria=ai_criteria, **shared)
    aligned = align_ai_with_rules(rules, ai)
    view = format_comparison(rules, aligned)
    ai_section = view["rating"].split("AI result", 1)[1]
    inclusion = view["rubric"].split("Inclusion and individual engagement", 1)[1].split("Tone and energy", 1)[0]

    assert "Excellent" not in ai_section
    assert "100.0" not in ai_section
    assert "Good" in ai_section
    assert "rules score kept" in inclusion
    assert "Not assessed" not in inclusion
    assert "words/min" in view["rubric"]
    assert "Measured frames: 3" in view["rubric"]


def test_ai_result_keeps_the_judge_score_when_it_differs_from_the_rules(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    video, transcript = _files(tmp_path)
    payload = _payload()
    for item in payload["criteria"]:
        if item["id"] == "inclusion":
            item["rating"] = None
            item["evidence"] = ""
    transport = ScriptedTransport([_http(json.dumps(payload))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    from class_quality.app import present_analysis

    error, _show_empty, show_results, rating, rubric, _findings, _metrics = present_analysis(
        str(video),
        str(transcript),
        client=client,
        measure=_measure([]),
        workspace_factory=lambda: str(tmp_path / "judge-score"),
        stills_fn=lambda *_args, **_kwargs: [],
        use_ai=True,
    )
    prompt = transport.calls[0]["messages"][0]["content"]
    rules_card, ai_card = rating.split("AI result", 1)
    rules_score = rules_card.split('cq-score">', 1)[1].split(" /", 1)[0]
    ai_score = ai_card.split('cq-score">', 1)[1].split(" /", 1)[0]

    assert error == ""
    assert show_results is True
    assert "Do not copy a rules rating." in prompt
    assert "rules score kept" not in rubric
    assert float(ai_score) <= float(rules_score) + 0.05
    assert "100.0" not in ai_card
    assert "Not scored" not in ai_card
    assert "missing rating was not treated as full marks" in rubric


def test_ai_card_does_not_repeat_the_rules_class_when_the_judge_scored_nothing():
    from class_quality.app import format_comparison
    from class_quality.rubric import CriterionResult, Report

    kept = "The judge did not return usable evidence, so this score is the rules score. "
    shared = dict(provisional=True, cap_applied=None, strengths=[], improvements=[], measurements={"duration_s": 600.0})
    criteria = [
        CriterionResult("opening", "Opening and rapport", 15, 1, "assessed", kept + "Rules opening.", []),
        CriterionResult("inclusion", "Inclusion and individual engagement", 20, 0, "assessed", kept + "Rules inclusion.", []),
        CriterionResult("tone", "Tone and energy", 15, None, "not_assessed", "Tone was not assessed.", []),
    ]
    rules = Report(
        classification="Poor",
        score=42.3,
        mode="rules_only",
        reliability_notes=[],
        criteria=criteria,
        **shared,
    )
    ai = Report(classification="Poor", score=42.3, mode="hybrid", reliability_notes=[], criteria=criteria, **shared)

    view = format_comparison(rules, ai)
    ai_section = view["rating"].split("AI result", 1)[1]

    assert "Not scored" in ai_section
    assert "42.3" not in ai_section
    assert "Poor" not in ai_section
    assert "matched the recording" in ai_section


def test_ai_card_hides_a_partial_100_when_the_judge_cannot_assign_a_class():
    from class_quality.app import format_comparison
    from class_quality.rubric import CriterionResult, Report

    shared = dict(provisional=True, cap_applied=None, strengths=[], improvements=[], measurements={"duration_s": 600.0})

    def item(criterion_id, name, weight, rating, status, explanation):
        return CriterionResult(criterion_id, name, weight, rating, status, explanation, [])

    rules = Report(
        classification="Poor",
        score=42.3,
        mode="rules_only",
        reliability_notes=[],
        criteria=[
            item("opening", "Opening and rapport", 15, 1, "assessed", "Rules opening."),
            item("inclusion", "Inclusion and individual engagement", 20, 0, "assessed", "Rules inclusion."),
            item("closure", "Lesson structure and closure", 10, 2, "assessed", "Rules closure."),
            item("homework", "Homework assigned", 10, 0, "assessed", "Rules homework."),
            item("technical", "Technical quality", 10, 2, "assessed", "Rules technical."),
        ],
        **shared,
    )
    ai = Report(
        classification=None,
        score=100.0,
        mode="hybrid",
        reliability_notes=["Insufficient evidence to assign a class."],
        criteria=[
            item("closure", "Lesson structure and closure", 10, 2, "assessed", "Judge closure."),
            item("homework", "Homework assigned", 10, 2, "assessed", "Judge homework."),
            item("technical", "Technical quality", 10, 2, "assessed", "Judge technical."),
            item("opening", "Opening and rapport", 15, None, "not_assessed", "Missing evidence."),
            item("inclusion", "Inclusion and individual engagement", 20, None, "not_assessed", "Missing evidence."),
        ],
        **shared,
    )

    view = format_comparison(rules, ai)
    rules_card, ai_card = view["rating"].split("AI result", 1)

    assert "Poor" in rules_card
    assert "42.3" in rules_card
    assert "Not enough evidence" in ai_card
    assert "100.0" not in ai_card
    assert "Class not assigned" not in ai_card
    assert "Judge closure." in view["rubric"]


def test_rules_page_shows_that_the_ai_judge_was_not_used():
    from class_quality.app import format_comparison
    from class_quality.rubric import CriterionResult, Report

    report = Report(
        classification=None,
        score=100.0,
        provisional=True,
        mode="rules_only",
        cap_applied=None,
        reliability_notes=["Insufficient evidence to assign a class."],
        criteria=[
            CriterionResult("technical", "Technical quality", 10, 2, "assessed", "Audio is present.", []),
        ],
        strengths=[],
        improvements=[],
        measurements={"duration_s": 120.0},
    )

    view = format_comparison(report, None)
    rendered = " ".join(view.values())

    assert "Rules result" in view["rating"]
    assert "AI result" in view["rating"]
    assert "not used" in view["rating"].lower()
    assert "Final result" not in view["rating"]
    assert "was not sent to Ollama Cloud" in view["rating"]
    assert "100.0" in view["rating"]
    assert "Class not assigned" in view["rating"]
    assert rendered.lower().count("100.0") == 1


def test_ai_switch_off_does_not_call_the_judge_when_cloud_is_permitted(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    video, transcript = _files(tmp_path)
    transport = ScriptedTransport([_http(json.dumps(_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    result = analyse_upload(
        str(video),
        str(transcript),
        client=client,
        measure=_measure([]),
        workspace_factory=lambda: str(tmp_path / "switch-off"),
        stills_fn=lambda *_args, **_kwargs: [],
        use_ai=False,
    )

    assert transport.calls == []
    assert "rules_only" in result
    assert "AI result" in result
    assert "not used" in result.lower()
    assert not (tmp_path / "switch-off").exists()


def test_ai_switch_on_uses_the_mocked_judge(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "1")
    video, transcript = _files(tmp_path)
    transport = ScriptedTransport([_http(json.dumps(_payload()))])
    client = OllamaCloudClient(api_key="test-key", transport=transport, permitted=True)

    result = analyse_upload(
        str(video),
        str(transcript),
        client=client,
        measure=_measure([]),
        workspace_factory=lambda: str(tmp_path / "switch-on"),
        stills_fn=lambda *_args, **_kwargs: [],
        use_ai=True,
    )

    assert len(transport.calls) == 1
    assert "hybrid" in result
    assert "Rules result" in result
    assert "rules result was given to the judge as context" in result.lower()
    assert "Final result" not in result
    for secret in SECRETS:
        assert secret not in result


def test_ai_switch_on_without_permission_does_not_call_the_judge(tmp_path: Path):
    video, transcript = _files(tmp_path)

    def forbid_client(*_args, **_kwargs):
        raise AssertionError("the judge must not be constructed")

    result = analyse_upload(
        str(video),
        str(transcript),
        client=forbid_client,
        measure=_measure([]),
        workspace_factory=lambda: str(tmp_path / "switch-blocked"),
        stills_fn=lambda *_args, **_kwargs: [],
        use_ai=True,
    )

    assert "rules_only" in result
    assert "not permitted" in result.lower()
    assert "was not called" in result.lower()
    assert not (tmp_path / "switch-blocked").exists()


def test_stage_tracker_highlights_the_current_step():
    from class_quality.app import _stages_html

    resting = _stages_html()
    current = _stages_html(3, 0.66, "Evaluating the class")
    done = _stages_html(5, 1, "Report ready")

    assert "cq-stage-active" not in resting
    assert current.count("cq-stage-done") == 2
    assert 'class="cq-stage-active"' in current
    assert "Evaluate" in current
    assert "66%" in current
    assert done.count("cq-stage-done") == 4
    assert "cq-stage-active" not in done


def test_live_progress_advances_past_the_old_midpoint_and_hides_the_last_result(monkeypatch):
    def slow(_video, _vtt, progress=None, **kwargs):
        on_step = kwargs["on_step"]
        on_step(1, 0.15, "Uploading the recording")
        on_step(3, 0.66, "Evaluating the class")
        time.sleep(1.1)
        return app_module._Outcome("Analysis failed: boom")

    monkeypatch.setattr(app_module, "_analyse", slow)
    frames = list(app_module._ui_event("clip.mp4", "clip.vtt", False))
    progress_frames = [frame[-1] for frame in frames if "cq-progress" in frame[-1]]
    percents = []
    for markup in progress_frames:
        percents.append(int(markup.split('aria-valuenow="', 1)[1].split('"', 1)[0]))

    assert any("Evaluating the class" in markup and "cq-stage-active" in markup for markup in progress_frames)
    assert any(percent > 66 for percent in percents)
    assert all(frame[1]["visible"] is False and frame[2]["visible"] is False for frame in frames[:-1])
    assert "Analysis failed" in frames[-1][0]
    assert "cq-progress" not in frames[-1][-1]


def test_failed_upload_does_not_invent_a_result():
    from class_quality.app import present_analysis

    error, show_empty, show_results, rating, rubric, findings, metrics = present_analysis(None, None)

    assert "mp4" in error.lower()
    assert show_empty is True
    assert show_results is False
    assert rating == "" and rubric == "" and findings == "" and metrics == ""
