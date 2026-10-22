"""Local transcription. Audio and text stay on this machine."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from class_quality.vtt import Cue, parse_vtt

# CPU int8 is the local default. A supplied transcript skips this model entirely.
DEFAULT_MODEL_SIZE = "base"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
SUPPORTED_LANGUAGES = {"en"}
MIN_LANGUAGE_PROBABILITY = 0.5


@dataclass(frozen=True)
class LocalTranscript:
    cues: list[Cue]
    reliable: bool
    reason: str
    vtt_text: str
    language: str | None = None
    language_probability: float | None = None
    speaker_labelled: bool = False


def transcript_from_segments(segments, language: str | None, probability: float | None) -> LocalTranscript:
    """Turn timed segments into cues through the existing WebVTT parser."""
    vtt_text = _render_vtt(segments)
    cues = parse_vtt(vtt_text)
    if not cues:
        return LocalTranscript(
            cues=[],
            reliable=False,
            reason="Local transcription did not produce a transcript.",
            vtt_text=vtt_text,
            language=language,
            language_probability=probability,
        )
    code = (language or "").lower()
    if code not in SUPPORTED_LANGUAGES:
        shown = language or "unknown"
        return LocalTranscript(
            cues=cues,
            reliable=False,
            reason=f"Transcription language {shown} is not supported, so conversational signals are unreliable.",
            vtt_text=vtt_text,
            language=language,
            language_probability=probability,
        )
    if probability is None or probability < MIN_LANGUAGE_PROBABILITY:
        return LocalTranscript(
            cues=cues,
            reliable=False,
            reason="Transcription language is uncertain, so conversational signals are unreliable.",
            vtt_text=vtt_text,
            language=language,
            language_probability=probability,
        )
    return LocalTranscript(
        cues=cues,
        reliable=True,
        reason="",
        vtt_text=vtt_text,
        language=language,
        language_probability=probability,
    )


def failed_transcript(reason: str = "Local transcription failed.") -> LocalTranscript:
    return LocalTranscript(cues=[], reliable=False, reason=reason, vtt_text="WEBVTT\n\n")


def transcribe_recording(path: str, *, loader=None, model_size: str = DEFAULT_MODEL_SIZE) -> LocalTranscript:
    """Transcribe one local file. An injected loader is used by tests and makes no network call."""
    load = loader or _load_model
    if loader is None and not _readable_audio(path):
        return failed_transcript("Local transcription failed. The recording audio could not be read.")
    try:
        model = load(model_size, DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE)
        segments, info = model.transcribe(str(path), vad_filter=True, word_timestamps=True)
        return transcript_from_segments(
            list(segments),
            getattr(info, "language", None),
            getattr(info, "language_probability", None),
        )
    except Exception:
        return failed_transcript()


def _load_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _readable_audio(path: str) -> bool:
    if shutil.which("ffprobe") is None:
        return False
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and '"codec_type": "audio"' in completed.stdout


def _render_vtt(segments) -> str:
    blocks = ["WEBVTT", ""]
    for segment in segments:
        text = str(getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        start, end = _bounds(segment)
        blocks.append(f"{_stamp(start)} --> {_stamp(end)}")
        blocks.append(f"SPEAKER: {text}")
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


def _bounds(segment) -> tuple[float, float]:
    start = float(segment.start)
    end = float(segment.end)
    timed: list[tuple[float, float]] = []
    for word in getattr(segment, "words", None) or []:
        word_start = getattr(word, "start", None)
        word_end = getattr(word, "end", None)
        if word_start is None or word_end is None:
            continue
        timed.append((float(word_start), float(word_end)))
    if timed:
        start = min(item[0] for item in timed)
        end = max(item[1] for item in timed)
    if end < start:
        end = start
    return start, end


def _stamp(seconds: float) -> str:
    millis = max(0, int(round(float(seconds) * 1000)))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
