"""Local media measurements through FFmpeg. Nothing is uploaded."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from class_quality.rubric import MediaStats

# Limited-range video uses Y=16 for black, so a cutoff of 16 would miss it.
DARK_LUMA = 20.0
STATIC_LUMA_DELTA = 1.0


class MediaError(RuntimeError):
    """FFmpeg or ffprobe could not measure the recording."""


@dataclass(frozen=True)
class ProbeInfo:
    duration_s: float
    video_present: bool
    audio_present: bool


@dataclass(frozen=True)
class AudioReport:
    loudness_lufs: float | None
    loudness_range_lu: float | None
    silence_s: float
    longest_silence_s: float


@dataclass(frozen=True)
class FrameReport:
    frame_count: int
    mean_luminance: float | None
    dark_frame_ratio: float | None
    static_frame_ratio: float | None


def parse_probe(payload: str) -> ProbeInfo:
    data = json.loads(payload)
    streams = data.get("streams") or []
    kinds = {stream.get("codec_type") for stream in streams}
    return ProbeInfo(
        duration_s=float(data["format"]["duration"]),
        video_present="video" in kinds,
        audio_present="audio" in kinds,
    )


def parse_audio_report(stderr: str) -> AudioReport:
    durations = [float(value) for value in re.findall(r"silence_duration:\s*([\d.]+)", stderr)]
    loudness = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", stderr)
    loudness_range = re.findall(r"LRA:\s+([\d.]+)\s+LU\b", stderr)
    return AudioReport(
        loudness_lufs=float(loudness[-1]) if loudness else None,
        loudness_range_lu=float(loudness_range[-1]) if loudness_range else None,
        silence_s=sum(durations),
        longest_silence_s=max(durations) if durations else 0.0,
    )


def parse_frame_report(stderr: str) -> FrameReport:
    frames: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for line in stderr.splitlines():
        if re.search(r"\bframe:\d+", line) and "pts_time" in line:
            if current:
                frames.append(current)
            current = {}
            continue
        if current is None:
            continue
        yavg = re.search(r"signalstats\.YAVG=([\d.]+)", line)
        ydif = re.search(r"signalstats\.YDIF=([\d.]+)", line)
        if yavg:
            current["yavg"] = float(yavg.group(1))
        if ydif:
            current["ydif"] = float(ydif.group(1))
    if current:
        frames.append(current)

    luminances = [frame["yavg"] for frame in frames if "yavg" in frame]
    # The first sample has no previous frame, so its difference is not a motion measure.
    deltas = [frame["ydif"] for frame in frames[1:] if "ydif" in frame]
    return FrameReport(
        frame_count=len(frames),
        mean_luminance=(sum(luminances) / len(luminances)) if luminances else None,
        dark_frame_ratio=(sum(value < DARK_LUMA for value in luminances) / len(luminances)) if luminances else None,
        static_frame_ratio=(sum(value < STATIC_LUMA_DELTA for value in deltas) / len(deltas)) if deltas else None,
    )


@dataclass(frozen=True)
class SampledFrame:
    """A still kept only in the analysis workspace. It is not an upload payload."""

    timestamp_s: float
    path: Path


def sample_frames(video_path: str, duration_s: float, workspace: Path) -> list[SampledFrame]:
    """Save three PNG stills beside the recording. Luminance is not a visual judgement."""
    if duration_s <= 0 or shutil.which("ffmpeg") is None:
        return []
    destination = Path(workspace)
    frames: list[SampledFrame] = []
    for fraction in (0.1, 0.5, 0.9):
        timestamp = float(duration_s) * fraction
        target = destination / f"frame-{len(frames):02d}.png"
        try:
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    str(target),
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode != 0 or not target.is_file() or target.stat().st_size <= 0:
            continue
        frames.append(SampledFrame(timestamp_s=timestamp, path=target))
    return frames


def default_runner(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.returncode, completed.stdout, completed.stderr


def _run(runner, command: list[str]) -> tuple[str, str]:
    code, stdout, stderr = runner(command)
    if code != 0:
        raise MediaError("ffmpeg failed")
    return stdout, stderr


def measure_recording(path: str, runner=None, frame_interval_s: int = 30) -> MediaStats:
    """Measure duration, loudness, silence, and sampled-frame statistics."""
    runner = runner or default_runner
    probe_stdout, _ = _run(
        runner,
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            path,
        ],
    )
    probe = parse_probe(probe_stdout)
    _, audio_stderr = _run(
        runner,
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            path,
            "-vn",
            "-af",
            "silencedetect=n=-40dB:d=5,ebur128",
            "-f",
            "null",
            "-",
        ],
    )
    audio = parse_audio_report(audio_stderr)
    _, frame_stderr = _run(
        runner,
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            path,
            "-vf",
            f"fps=1/{frame_interval_s},signalstats,metadata=print",
            "-an",
            "-f",
            "null",
            "-",
        ],
    )
    frames = parse_frame_report(frame_stderr)
    return MediaStats(
        duration_s=probe.duration_s,
        audio_present=probe.audio_present,
        video_present=probe.video_present,
        loudness_lufs=audio.loudness_lufs,
        loudness_range_lu=audio.loudness_range_lu,
        silence_s=audio.silence_s,
        longest_silence_s=audio.longest_silence_s,
        mean_luminance=frames.mean_luminance,
        dark_frame_ratio=frames.dark_frame_ratio,
        static_frame_ratio=frames.static_frame_ratio,
        frame_count=frames.frame_count,
    )
