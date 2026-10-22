"""Local transcript and frame evidence. No dataset files and no live cloud calls."""

import socket
import urllib.request
from pathlib import Path

import pytest

from class_quality.analyze import analyze
from class_quality.app import analyse_upload, format_dashboard
from class_quality.judge import FrameStill, apply_judgement
from class_quality.media import MediaError, sample_frames
from class_quality.rubric import MediaStats
from class_quality.transcribe import transcript_from_segments, transcribe_recording
from tests.test_app import MEDIA, Recorder, ScriptedTransport, VTT, _files, _measure
from tests.test_judge import _item, _pack

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def cloud_permission_stays_off(monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)


class Segment:
    def __init__(self, start, end, text, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []


class Word:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word


def _english_segments():
    return [
        Segment(10.0, 12.0, "Hello class"),
        Segment(40.0, 42.0, "Let us continue"),
        Segment(50.0, 52.0, "Look at this example"),
        Segment(80.0, 84.0, "Please do the homework tonight"),
        Segment(500.0, 504.0, "See you next class"),
    ]


def _block_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


def test_segment_timestamps_survive_the_existing_vtt_parser():
    cues = transcript_from_segments([Segment(1.5, 3.25, "Hello class")], language="en", probability=0.9).cues

    assert len(cues) == 1
    assert cues[0].start_s == pytest.approx(1.5)
    assert cues[0].end_s == pytest.approx(3.25)
    assert cues[0].text == "Hello class"


def test_word_timestamps_bound_the_segment_without_splitting_the_phrase():
    words = [Word(1.25, 1.5, "Hello"), Word(1.5, 2.0, "class")]
    cues = transcript_from_segments(
        [Segment(0.0, 5.0, "Hello class", words)],
        language="en",
        probability=0.9,
    ).cues

    assert cues[0].start_s == pytest.approx(1.25)
    assert cues[0].end_s == pytest.approx(2.0)
    assert cues[0].text == "Hello class"


def test_unsupported_or_uncertain_language_is_not_reliable():
    unsupported = transcript_from_segments([Segment(0.0, 2.0, "Hello class")], language="ml", probability=0.99)
    uncertain = transcript_from_segments([Segment(0.0, 2.0, "Hello class")], language="en", probability=0.2)
    missing = transcript_from_segments([], language="en", probability=0.9)

    assert unsupported.reliable is False
    assert "language" in unsupported.reason.lower()
    assert uncertain.reliable is False
    assert missing.reliable is False
    assert missing.cues == []


def test_transcription_failure_stays_local_and_does_not_open_a_socket(monkeypatch):
    _block_network(monkeypatch)

    def loader(*_args, **_kwargs):
        raise RuntimeError("model missing")

    result = transcribe_recording("clip.mp4", loader=loader)

    assert result.reliable is False
    assert result.cues == []
    assert "fail" in result.reason.lower()


def test_mocked_local_transcription_preserves_timestamps_without_a_network_call(monkeypatch):
    _block_network(monkeypatch)

    class Info:
        language = "en"
        language_probability = 0.91

    class Model:
        def transcribe(self, path, **kwargs):
            assert path == "clip.mp4"
            assert kwargs.get("vad_filter") is True
            return iter([Segment(8.0, 9.5, "Hello everyone")]), Info()

    result = transcribe_recording("clip.mp4", loader=lambda *_args, **_kwargs: Model())

    assert result.reliable is True
    assert result.language == "en"
    assert result.cues[0].start_s == pytest.approx(8.0)
    assert result.cues[0].end_s == pytest.approx(9.5)


def test_mp4_without_vtt_uses_local_transcription_and_can_assign_a_class(tmp_path: Path, monkeypatch):
    _block_network(monkeypatch)
    calls = []

    def transcribe(path):
        calls.append(path)
        return transcript_from_segments(_english_segments(), language="en", probability=0.95)

    report = analyze(
        str(tmp_path / "clip.mp4"),
        vtt_path=None,
        measure=lambda _path, **_kwargs: MEDIA,
        transcribe=transcribe,
    )

    assert calls
    homework = next(item for item in report.criteria if item.id == "homework")
    inclusion = next(item for item in report.criteria if item.id == "inclusion")
    assert homework.status == "assessed"
    assert homework.rating == 2
    assert "00:01:20" in homework.timestamps
    assert inclusion.status == "not_assessed"
    assert report.classification is not None
    assert any("speaker" in note.lower() for note in report.reliability_notes)
    assert report.measurements["transcript_source"] == "local"
    assert report.measurements["detected_language"] == "en"
    assert report.measurements["language_probability"] == pytest.approx(0.95)


def test_supplied_vtt_is_used_and_the_recording_is_not_transcribed(tmp_path: Path):
    vtt = tmp_path / "session.vtt"
    vtt.write_text(VTT, encoding="utf-8")

    def transcribe(_path):
        raise AssertionError("a supplied transcript must not be replaced")

    report = analyze(
        str(tmp_path / "clip.mp4"),
        vtt_path=str(vtt),
        measure=lambda _path, **_kwargs: MEDIA,
        transcribe=transcribe,
    )

    assert report.mode == "rules_only"
    assert any(item.id == "homework" and item.rating == 2 for item in report.criteria)


def test_unreliable_transcription_leaves_conversation_unassessed_and_the_class_unassigned(tmp_path: Path):
    def transcribe(_path):
        return transcript_from_segments([Segment(1.0, 2.0, "Hello class")], language="ml", probability=0.99)

    report = analyze(
        str(tmp_path / "clip.mp4"),
        measure=lambda _path, **_kwargs: MEDIA,
        transcribe=transcribe,
    )

    conversational = [item for item in report.criteria if item.id in {"opening", "inclusion", "closure", "homework"}]
    assert conversational
    assert all(item.status == "not_assessed" and item.rating is None for item in conversational)
    assert report.classification is None
    assert any("language" in note.lower() for note in report.reliability_notes)


def test_transcription_failure_explains_the_limit_and_deletes_temporary_files(tmp_path: Path):
    workspace = tmp_path / "job"

    def factory() -> str:
        workspace.mkdir()
        return str(workspace)

    def transcribe(_path):
        (workspace / "audio.wav").write_bytes(b"local-audio")
        raise RuntimeError("decoder failed")

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic-mp4")
    result = analyse_upload(
        str(video),
        None,
        measure=lambda _path, **_kwargs: MEDIA,
        workspace_factory=factory,
        transcribe=transcribe,
        sample_frames=lambda *_args, **_kwargs: [],
    )

    assert "fail" in result.lower()
    assert "not assigned" in result.lower()
    assert "Analysis failed" not in result
    assert not workspace.exists()


def test_upload_without_vtt_shows_transcription_progress_and_cleans_the_transcript(tmp_path: Path, monkeypatch):
    _block_network(monkeypatch)
    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic-mp4")
    workspace = tmp_path / "job"
    progress = Recorder()
    seen = []

    def factory() -> str:
        workspace.mkdir()
        return str(workspace)

    def transcribe(_path):
        return transcript_from_segments(_english_segments(), language="en", probability=0.95)

    def measure(path, **_kwargs):
        seen.append(path)
        assert (workspace / "transcript.vtt").is_file()
        assert "Hello class" in (workspace / "transcript.vtt").read_text(encoding="utf-8")
        return MEDIA

    result = analyse_upload(
        str(video),
        None,
        progress=progress,
        measure=measure,
        workspace_factory=factory,
        transcribe=transcribe,
        sample_frames=lambda *_args, **_kwargs: [],
    )

    assert any(desc and "Transcribing the recording locally" in desc for _fraction, desc in progress.events)
    assert "rules_only" in result
    assert "AI-Assisted" not in result
    assert os_permission_stays_off()
    assert seen
    assert not workspace.exists()


def test_upload_with_vtt_does_not_transcribe(tmp_path: Path):
    video, transcript = _files(tmp_path)

    def transcribe(_path):
        raise AssertionError("supplied vtt")

    result = analyse_upload(
        str(video),
        str(transcript),
        measure=_measure([]),
        transcribe=transcribe,
        sample_frames=lambda *_args, **_kwargs: [],
        stills_fn=lambda *_args, **_kwargs: [],
    )

    assert "rules_only" in result


def test_sampled_frames_record_timestamps_and_are_real_files(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _write_synthetic_video(video)
    frames = sample_frames(str(video), 1.0, tmp_path)

    assert len(frames) == 3
    assert [round(frame.timestamp_s, 1) for frame in frames] == [0.1, 0.5, 0.9]
    for frame in frames:
        assert frame.path.is_file()
        assert frame.path.stat().st_size > 0
        assert frame.path.suffix == ".png"
        assert not hasattr(frame, "image_base64") or getattr(frame, "image_base64", None) in (None, "")


def test_frame_files_are_removed_when_analysis_fails(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic-mp4")
    _ignored, transcript = _files(tmp_path)
    workspace = tmp_path / "frames-job"
    created: list[Path] = []

    def factory() -> str:
        workspace.mkdir()
        return str(workspace)

    def frames(_video, _duration, workspace_path):
        target = Path(workspace_path) / "frame-00.png"
        target.write_bytes(b"png")
        created.append(target)
        from class_quality.media import SampledFrame

        return [SampledFrame(timestamp_s=60.0, path=target)]

    def measure(_path, **_kwargs):
        assert created and created[0].is_file()
        raise MediaError("ffmpeg failed")

    analyse_upload(
        str(video),
        str(transcript),
        measure=measure,
        workspace_factory=factory,
        sample_frames=frames,
    )

    assert created and not created[0].exists()
    assert not workspace.exists()


def test_sampled_frames_do_not_invent_a_visual_rating():
    report = apply_judgement(
        _pack(frames=[FrameStill(timestamp_s=60.0, image_base64=None)]),
        {"criteria": [_item("teacher_presence", 2, "00:01:00", "A person is visible")]},
    )
    presence = next(item for item in report.criteria if item.id == "teacher_presence")

    assert presence.status == "not_assessed"
    assert presence.rating is None
    assert "luminance" not in presence.explanation.lower()
    assert "no still images were available" not in presence.explanation.lower()
    assert "blur" in presence.explanation.lower()


def test_a_judge_attempt_without_usable_scores_is_not_labelled_ai_assisted():
    report = apply_judgement(_pack(), {"criteria": []})
    rating = format_dashboard(report)["rating"]

    assert report.mode == "rules_only"
    assert report.classification is None
    assert "AI-Assisted" not in rating
    assert "Provisional" in rating
    assert any("usable" in note.lower() for note in report.reliability_notes)
    assert all(item.rating is None or item.status == "assessed" for item in report.criteria)


def os_permission_stays_off() -> bool:
    import os

    return os.environ.get("OLLAMA_CLOUD_PERMITTED") != "1"


def _write_synthetic_video(path: Path) -> None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not available")
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=160x120:rate=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not path.is_file():
        pytest.skip("ffmpeg could not create a synthetic clip")
