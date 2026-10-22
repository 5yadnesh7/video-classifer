"""End-to-end rules-only analysis on synthetic inputs. No network and no dataset files."""

import json
import socket
import urllib.request
from dataclasses import asdict
from pathlib import Path

import pytest

from class_quality.analyze import analyze
from class_quality.media import MediaError
from class_quality.rubric import MediaStats

VTT = """WEBVTT

1
00:00:30.000 --> 00:00:32.000
Pat Smith: Hello, how are you?

2
00:00:40.000 --> 00:00:42.000
Quinn Jones: I am fine.

3
00:00:50.000 --> 00:00:52.000
Rae Stone: Hello.

4
00:01:00.000 --> 00:01:02.000
Pat Smith: Welcome, please introduce yourself?

5
00:01:20.000 --> 00:01:22.000
Quinn Jones: I am here.

6
00:02:00.000 --> 00:02:02.000
Rae Stone: My answer.

7
00:03:00.000 --> 00:03:02.000
Quinn Jones: Another answer.

8
00:04:00.000 --> 00:04:02.000
Quinn Jones: One more answer.

9
00:08:00.000 --> 00:08:02.000
Pat Smith: What is the next move?

10
00:08:10.000 --> 00:08:12.000
Quinn Jones: This one.

11
00:08:20.000 --> 00:08:22.000
Pat Smith: Please do the homework tonight.

12
00:09:20.000 --> 00:09:22.000
Pat Smith: See you next class, bye.
"""

MEDIA = MediaStats(
    duration_s=600.0,
    audio_present=True,
    video_present=True,
    loudness_lufs=-18.0,
    loudness_range_lu=8.0,
    silence_s=20.0,
    longest_silence_s=6.0,
    mean_luminance=90.0,
    dark_frame_ratio=0.0,
    static_frame_ratio=0.1,
    frame_count=20,
)


def _dump(report) -> str:
    return json.dumps(asdict(report))


def test_analyze_pseudonymises_names_and_ignores_the_filename(tmp_path: Path, caplog):
    vtt = tmp_path / "session.vtt"
    vtt.write_text(VTT, encoding="utf-8")

    def measure(path: str, **kwargs):
        assert path.endswith("excellent_secret.mp4") or path.endswith("poor_secret.mp4")
        return MEDIA

    caplog.set_level("DEBUG")
    first = analyze(str(tmp_path / "excellent_secret.mp4"), vtt_path=str(vtt), measure=measure)
    second = analyze(str(tmp_path / "poor_secret.mp4"), vtt_path=str(vtt), measure=measure)

    assert first.classification == second.classification == "Excellent"
    assert first.mode == "rules_only"
    assert first.provisional is True
    rendered = _dump(first) + _dump(second) + caplog.text
    for secret in ("Pat", "Smith", "Quinn", "Jones", "Rae", "Stone"):
        assert secret not in rendered


def test_missing_transcript_does_not_invent_a_class(tmp_path: Path):
    report = analyze(str(tmp_path / "clip.mp4"), vtt_path=None, measure=lambda path, **kwargs: MEDIA)

    assert report.classification is None
    assert any("no transcript" in note.lower() for note in report.reliability_notes)
    assert any(item.id == "homework" and item.status == "not_assessed" for item in report.criteria)


def test_ffmpeg_failure_still_returns_a_transcript_report_and_deletes_the_workspace(tmp_path: Path):
    vtt = tmp_path / "session.vtt"
    vtt.write_text(VTT, encoding="utf-8")
    workspace = tmp_path / "job"

    def factory() -> str:
        workspace.mkdir()
        return str(workspace)

    def measure(path: str, **kwargs):
        (workspace / "partial.txt").write_text("Pat Smith", encoding="utf-8")
        raise MediaError("ffmpeg failed")

    report = analyze(
        str(tmp_path / "clip.mp4"),
        vtt_path=str(vtt),
        measure=measure,
        workspace_factory=factory,
    )

    assert not workspace.exists()
    assert report.classification == "Excellent"
    assert any(item.id == "technical" and item.status == "not_assessed" for item in report.criteria)
    assert "Pat Smith" not in _dump(report)
    assert any("ffmpeg" in note.lower() for note in report.reliability_notes)


def test_a_supplied_but_missing_vtt_path_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        analyze(str(tmp_path / "clip.mp4"), vtt_path=str(tmp_path / "missing.vtt"), measure=lambda path, **kwargs: MEDIA)


def test_rules_only_analysis_does_not_touch_the_network(tmp_path: Path, monkeypatch):
    vtt = tmp_path / "session.vtt"
    vtt.write_text(VTT, encoding="utf-8")

    def blocked(*args, **kwargs):
        raise AssertionError("network call")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    report = analyze(str(tmp_path / "clip.mp4"), vtt_path=str(vtt), measure=lambda path, **kwargs: MEDIA)
    assert report.classification == "Excellent"
