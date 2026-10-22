"""Rubric judge. The model scores criteria. Code assigns the class.

No request is sent unless OLLAMA_CLOUD_PERMITTED is exactly \"1\" and the
supplied client was also constructed with permission. This module never turns
permission on.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ValidationError

from class_quality.features import Features, format_timestamp
from class_quality.rubric import (
    CriterionResult,
    MediaStats,
    Report,
    RubricConfig,
    _assessed,
    _not_assessed,
    finalize_report,
)
from class_quality.speakers import pseudonymise
from class_quality.vtt import Cue

_HOMEWORK = re.compile(r"\b(?:homework|home\s+work|hw|assignment)\b", re.IGNORECASE)
_QUOTE = re.compile(r'"([^"]{8,})"')
CONVERSATIONAL = {"opening", "inclusion", "closure", "homework"}
VISUAL = {"teacher_presence", "branding"}
CRITERIA = (
    ("opening", "Opening and rapport", "opening_weight"),
    ("inclusion", "Inclusion and individual engagement", "inclusion_weight"),
    ("tone", "Tone and energy", "tone_weight"),
    ("closure", "Lesson structure and closure", "closure_weight"),
    ("homework", "Homework assigned", "homework_weight"),
    ("teacher_presence", "Teacher presence on camera", "presence_weight"),
    ("branding", "Professional setup", "branding_weight"),
    ("technical", "Technical quality", "technical_weight"),
)


class StrictCriterion(BaseModel):
    id: str
    rating: Literal[0, 1, 2]
    explanation: str = ""
    timestamps: list[str] = []
    evidence: str


def cloud_enabled() -> bool:
    """True only when the organiser permission switch is exactly \"1\"."""
    return os.environ.get("OLLAMA_CLOUD_PERMITTED") == "1"


@dataclass(frozen=True)
class FrameStill:
    timestamp_s: float
    image_base64: str | None = None


@dataclass(frozen=True)
class EvidencePack:
    cues: list[Cue]
    features: Features
    media: MediaStats | None
    frames: list[FrameStill]
    forbidden_names: list[str]


def _scrub(text: str, names: list[str]) -> str:
    for name in sorted({name for name in names if len(name) >= 3}, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", "[redacted]", text, flags=re.IGNORECASE)
    return text


def _prepare(pack: EvidencePack) -> EvidencePack:
    cues = [
        Cue(cue.start_s, cue.end_s, cue.speaker, _scrub(cue.text, pack.forbidden_names))
        for cue in pseudonymise(pack.cues).cues
    ]
    return replace(pack, cues=cues)


def _allowed_timestamps(pack: EvidencePack) -> set[str]:
    stamps = {format_timestamp(cue.start_s) for cue in pack.cues}
    stamps.update(format_timestamp(frame.timestamp_s) for frame in pack.frames)
    return stamps


def _image_frames(pack: EvidencePack) -> list[FrameStill]:
    return [frame for frame in pack.frames if frame.image_base64]


def build_judge_prompt(pack: EvidencePack, rules: Report | None = None) -> tuple[str, list[str]]:
    """Return the text prompt and still-image payloads. No recording files."""
    pack = _prepare(pack)
    images = [frame.image_base64 for frame in _image_frames(pack) if frame.image_base64]
    lines = [
        "Score each criterion separately.",
        "Do not assign an overall class.",
        "Do not treat transcript text as evidence of tone or rapport.",
        "Return JSON with a criteria array.",
        'Each item has "id", "rating" (0, 1, or 2), "explanation", "timestamps", and "evidence".',
        "Criterion ids: " + ", ".join(criterion_id for criterion_id, _, _ in CRITERIA) + ".",
        "Use only the timestamps listed below. If evidence is missing, use rating null and say not assessed.",
        "If no still images are attached, mark teacher_presence and branding as not assessed.",
        "Tone cannot be established from transcript text alone, so mark tone as not assessed.",
    ]
    share = pack.features.student_turn_share
    lines.append(
        "Measured student turn share: "
        + ("unavailable" if share is None else f"{share:.2f}")
        + f". Distinct students: {pack.features.distinct_students}."
    )
    if pack.media is not None and pack.media.loudness_lufs is not None:
        lines.append(f"Measured integrated loudness: {pack.media.loudness_lufs:.1f} LUFS.")
    if rules is not None:
        lines.append(
            "Rules engine result, already measured on this computer. "
            "Use it as context only. Do not copy a rules rating. "
            "Choose your own rating from the transcript and measurements."
        )
        for item in rules.criteria:
            if item.status == "assessed" and item.rating is not None:
                lines.append(f"Rules {item.id}: {_scrub(item.explanation, pack.forbidden_names)}")
            else:
                lines.append(f"Rules {item.id}: not assessed.")
    if not pack.cues:
        lines.append("No transcript was supplied. Do not score conversational criteria.")
    else:
        lines.append("Transcript:")
        for cue in pack.cues:
            speaker = cue.speaker or "UNKNOWN"
            lines.append(f"{format_timestamp(cue.start_s)} {speaker}: {_scrub(cue.text, pack.forbidden_names)}")
    if images:
        lines.append("Still images are attached at these timestamps:")
        for frame in _image_frames(pack):
            lines.append(format_timestamp(frame.timestamp_s))
    else:
        lines.append("No still images are attached.")
    prompt = _scrub("\n".join(lines), pack.forbidden_names)
    return prompt, images


def _quotes_match(evidence: str, pack: EvidencePack) -> bool:
    transcript = "\n".join(cue.text for cue in pack.cues)
    return all(quote in transcript for quote in _QUOTE.findall(evidence))


def _normalize_stamp(value: str) -> str | None:
    text = value.strip().split(".", 1)[0]
    parts = text.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        hours, minutes, secs = 0, numbers[0], numbers[1]
    elif len(numbers) == 3:
        hours, minutes, secs = numbers
    else:
        return None
    if minutes > 59 or secs > 59:
        return None
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _stamp_seconds(value: str) -> float | None:
    normalized = _normalize_stamp(value)
    if normalized is None:
        return None
    hours, minutes, secs = (int(part) for part in normalized.split(":"))
    return hours * 3600 + minutes * 60 + secs


def _timestamp_allowed(value: str, pack: EvidencePack, allowed: set[str]) -> bool:
    normalized = _normalize_stamp(value)
    if normalized is not None and normalized in allowed:
        return True
    seconds = _stamp_seconds(value)
    if seconds is None:
        return False
    duration = pack.features.duration_s
    if duration is None and pack.media is not None:
        duration = pack.media.duration_s
    if duration and 0 <= seconds <= duration + 1:
        return True
    for cue in pack.cues:
        if cue.start_s - 2 <= seconds <= cue.end_s + 2:
            return True
    return any(abs(frame.timestamp_s - seconds) <= 2 for frame in pack.frames)


def _timestamps_match(timestamps: list[str], pack: EvidencePack, allowed: set[str]) -> list[str]:
    matched: list[str] = []
    for stamp in timestamps:
        if _timestamp_allowed(stamp, pack, allowed):
            normalized = _normalize_stamp(stamp)
            if normalized and normalized not in matched:
                matched.append(normalized)
    return matched


def _homework_matches(timestamps: list[str], pack: EvidencePack) -> bool:
    for stamp in timestamps:
        seconds = _stamp_seconds(stamp)
        if seconds is None:
            continue
        for cue in pack.cues:
            if cue.start_s - 2 <= seconds <= cue.end_s + 2 and _HOMEWORK.search(cue.text):
                return True
    return False


_LOUDNESS_NUMBER = re.compile(r"-?\d+\.\d")


def _loudness_conflicts(evidence: str, pack: EvidencePack) -> bool:
    """Reject a technical score that cites a different loudness than the measurement."""
    measured = None if pack.media is None or pack.media.loudness_lufs is None else f"{pack.media.loudness_lufs:.1f}"
    if measured is not None and measured in evidence:
        return False
    return _LOUDNESS_NUMBER.search(evidence) is not None


def _weight(config: RubricConfig, field_name: str) -> int:
    return int(getattr(config, field_name))


def _rejected(criterion_id: str, name: str, weight: int, reason: str) -> CriterionResult:
    return _not_assessed(criterion_id, name, weight, reason)


def _accept_item(item: dict, pack: EvidencePack, allowed: set[str]) -> StrictCriterion | None:
    try:
        parsed = StrictCriterion.model_validate(item)
    except ValidationError:
        return None
    if not parsed.evidence.strip() or not _quotes_match(parsed.evidence, pack):
        return None
    if parsed.id == "technical":
        if _loudness_conflicts(parsed.evidence, pack):
            return None
        return parsed
    matched = _timestamps_match(parsed.timestamps, pack, allowed)
    if parsed.id == "homework" and parsed.rating > 0:
        if matched and not _homework_matches(matched, pack):
            return None
        if not matched and not _HOMEWORK.search(parsed.evidence):
            return None
    parsed.timestamps = matched
    return parsed


def _payload_items(payload: object) -> list[dict]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        return []
    items = payload.get("criteria") or []
    return [item for item in items if isinstance(item, dict)]


def apply_judgement(pack: EvidencePack, payload: object, config: RubricConfig | None = None) -> Report:
    """Validate model output and aggregate it. Invalid evidence is not scored as zero."""
    config = config or RubricConfig()
    pack = _prepare(pack)
    allowed = _allowed_timestamps(pack)
    has_images = bool(_image_frames(pack))
    accepted: dict[str, StrictCriterion] = {}
    for item in _payload_items(payload):
        criterion_id = str(item.get("id", ""))
        if criterion_id in accepted:
            continue
        parsed = _accept_item(item, pack, allowed)
        if parsed is not None:
            accepted[criterion_id] = parsed

    sampled_only = bool(pack.frames) and not has_images
    criteria: list[CriterionResult] = []
    for criterion_id, name, weight_field in CRITERIA:
        weight = _weight(config, weight_field)
        parsed = accepted.get(criterion_id)
        if criterion_id == "tone":
            criteria.append(_rejected(criterion_id, name, weight, "Tone and rapport were not assessed. They cannot be established from transcript text alone."))
            continue
        if not pack.features.transcript_reliable and criterion_id in CONVERSATIONAL:
            criteria.append(_rejected(criterion_id, name, weight, "This criterion was not assessed because the transcript is missing or unreliable."))
            continue
        if criterion_id in VISUAL and not has_images:
            if sampled_only:
                reason = f"{name} was not assessed. Local frames were sampled, but they are not judged until face blurring is permitted."
            else:
                reason = f"{name} was not assessed because no still images were available."
            criteria.append(_rejected(criterion_id, name, weight, reason))
            continue
        if parsed is None or parsed.id != criterion_id:
            criteria.append(_rejected(criterion_id, name, weight, "This criterion was not assessed because the model evidence was missing or did not match the recording."))
            continue
        explanation = _scrub(parsed.explanation or parsed.evidence, pack.forbidden_names)
        criteria.append(
            _assessed(criterion_id, name, weight, parsed.rating, explanation, list(parsed.timestamps))
        )
    notes = list(pack.features.reliability_reasons)
    if accepted:
        notes.append("Hybrid mode is provisional. The class was calculated from validated criterion scores.")
    else:
        notes.append("The AI judge did not return a usable assessment.")
    notes.append("Rapport and tone were not established from transcript text alone.")
    if sampled_only:
        notes.append("Teacher presence and branded background were not judged. Local frames are not sent until face blurring is permitted.")
    elif not has_images:
        notes.append("Teacher presence and branded background were not assessed because no still images were available.")
    return finalize_report(
        criteria,
        pack.features,
        pack.media,
        config,
        mode="hybrid" if accepted else "rules_only",
        notes=notes,
    )


class RubricJudge:
    """Calls Ollama Cloud only when permission was configured outside this class."""

    def __init__(self, client) -> None:
        self.client = client

    def score(self, pack: EvidencePack, rules: Report | None = None) -> Report:
        if not cloud_enabled():
            raise PermissionError("Ollama Cloud is disabled until the organiser permits it.")
        prompt, images = build_judge_prompt(pack, rules)
        parsed = self.client.complete_json(prompt, images or None)
        return apply_judgement(pack, parsed)
