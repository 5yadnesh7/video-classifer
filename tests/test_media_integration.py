"""Integration with the system FFmpeg. Synthetic clips only, deleted afterwards."""

import shutil
import subprocess
from pathlib import Path

import pytest

from class_quality.media import MediaError, measure_recording

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg is not installed")


def _render(path: Path, video: str, audio: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            video,
            "-f",
            "lavfi",
            "-i",
            audio,
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_measure_recording_reads_a_synthetic_clip_and_leaves_no_extra_files(tmp_path: Path):
    clip = tmp_path / "tone.mp4"
    _render(clip, "color=c=gray:s=160x90:d=2:r=10", "sine=frequency=440:duration=2")
    before = set(tmp_path.iterdir())

    stats = measure_recording(str(clip), frame_interval_s=1)

    assert stats.duration_s == pytest.approx(2.0, abs=0.2)
    assert stats.audio_present is True
    assert stats.video_present is True
    assert stats.loudness_lufs is not None
    assert stats.loudness_lufs > -40
    assert stats.frame_count >= 1
    assert stats.mean_luminance is not None
    assert stats.mean_luminance > 50
    assert set(tmp_path.iterdir()) == before


def test_silence_in_a_synthetic_clip_is_measured(tmp_path: Path):
    clip = tmp_path / "quiet.mp4"
    _render(clip, "color=c=black:s=160x90:d=6:r=5", "anullsrc=r=44100:cl=mono:d=6")

    stats = measure_recording(str(clip), frame_interval_s=2)

    assert stats.silence_s > 1
    assert stats.dark_frame_ratio == pytest.approx(1.0)


def test_missing_recording_raises_media_error(tmp_path: Path):
    with pytest.raises(MediaError):
        measure_recording(str(tmp_path / "missing.mp4"))
