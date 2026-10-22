"""Rules-only analysis. This module does not make network calls."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from class_quality.features import extract_features, format_timestamp
from class_quality.media import MediaError, measure_recording
from class_quality.rubric import Report, score_rules
from class_quality.speakers import pseudonymise
from class_quality.transcribe import LocalTranscript, failed_transcript
from class_quality.vtt import parse_vtt

logger = logging.getLogger(__name__)


def analyze(
    video_path: str,
    vtt_path: str | None = None,
    measure=measure_recording,
    workspace_factory=tempfile.mkdtemp,
    transcribe=None,
    local_transcript: LocalTranscript | None = None,
    sample_frames=None,
) -> Report:
    """Score one recording from local evidence. Real names stay inside this function."""
    workspace = Path(workspace_factory())
    try:
        (workspace / "scratch.txt").write_text("workspace", encoding="utf-8")
        sampled = _sample(sample_frames, video_path, 0.0, workspace)
        media = None
        measurement_failed = False
        try:
            media = measure(video_path)
        except MediaError:
            measurement_failed = True
        if sample_frames is not None and media is not None and not sampled:
            sampled = _sample(sample_frames, video_path, media.duration_s, workspace)
        local, cues = _cues(video_path, vtt_path, local_transcript, transcribe)
        pseudonymised = pseudonymise(cues)
        duration = media.duration_s if media is not None else None
        features = extract_features(pseudonymised.cues, duration_s=duration)
        if local is not None and not local.reliable:
            features = replace(
                features,
                transcript_reliable=False,
                reliability_reasons=[local.reason, *features.reliability_reasons] if local.reason else features.reliability_reasons,
            )
        report = score_rules(features, media)
        extra: list[str] = []
        if measurement_failed:
            extra.append("FFmpeg could not measure the recording. Technical quality was not assessed.")
        if local is not None and local.cues and not local.speaker_labelled:
            extra.append("Local transcription has no speaker labels. Student participation was not measured.")
        if sampled:
            stamps = ", ".join(format_timestamp(frame.timestamp_s) for frame in sampled)
            extra.append(
                "Local frames were sampled at "
                + stamps
                + ". Teacher presence and branded background were not judged from those frames or from luminance."
            )
        if extra:
            report = replace(report, reliability_notes=[*extra, *report.reliability_notes])
        report = attach_transcript_metadata(report, local, supplied=vtt_path is not None)
        logger.debug("rules-only analysis finished")
        return report
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def attach_transcript_metadata(report: Report, local: LocalTranscript | None, *, supplied: bool) -> Report:
    """Record where the transcript came from. This does not change the score."""
    measurements = dict(report.measurements)
    if supplied:
        measurements["transcript_source"] = "supplied"
    elif local is not None:
        measurements["transcript_source"] = "local"
        if local.language:
            measurements["detected_language"] = local.language
        if isinstance(local.language_probability, (int, float)):
            measurements["language_probability"] = float(local.language_probability)
    else:
        measurements["transcript_source"] = "missing"
    return replace(report, measurements=measurements)


def _sample(sample_frames, video_path: str, duration_s: float, workspace: Path):
    if sample_frames is None:
        return []
    try:
        return list(sample_frames(video_path, duration_s, workspace) or [])
    except Exception:
        logger.debug("local frame sampling failed")
        return []


def _cues(video_path: str, vtt_path: str | None, local_transcript, transcribe):
    if vtt_path is not None:
        transcript_path = Path(vtt_path)
        if not transcript_path.is_file():
            raise FileNotFoundError(vtt_path)
        return None, parse_vtt(transcript_path.read_text(encoding="utf-8"))
    local = local_transcript
    if local is None and transcribe is not None:
        try:
            local = transcribe(video_path)
        except Exception:
            local = failed_transcript()
    if local is None:
        return None, []
    return local, list(local.cues)
