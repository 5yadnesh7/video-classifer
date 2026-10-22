"""Transparent rules-only rubric. Weights are configurable and not fitted to filenames."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from class_quality.features import Features

SEVERE_LOUDNESS_LUFS = -35.0


@dataclass(frozen=True)
class RubricConfig:
    opening_weight: int = 15
    inclusion_weight: int = 20
    tone_weight: int = 15
    closure_weight: int = 10
    homework_weight: int = 10
    presence_weight: int = 15
    branding_weight: int = 5
    technical_weight: int = 10
    excellent_min: float = 80.0
    good_min: float = 65.0
    average_min: float = 50.0
    min_assessed_weight: int = 40
    inclusion_high: float = 0.50
    inclusion_mid: float = 0.40


@dataclass(frozen=True)
class MediaStats:
    duration_s: float
    audio_present: bool
    video_present: bool
    loudness_lufs: float | None
    loudness_range_lu: float | None
    silence_s: float
    longest_silence_s: float
    mean_luminance: float | None
    dark_frame_ratio: float | None
    static_frame_ratio: float | None
    frame_count: int


@dataclass(frozen=True)
class CriterionResult:
    id: str
    name: str
    weight: int
    rating: int | None
    status: str
    explanation: str
    timestamps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Report:
    classification: str | None
    score: float | None
    provisional: bool
    mode: str
    cap_applied: str | None
    reliability_notes: list[str]
    criteria: list[CriterionResult]
    strengths: list[str]
    improvements: list[str]
    measurements: dict[str, object]


def classify_score(score: float, config: RubricConfig | None = None) -> str:
    config = config or RubricConfig()
    if score >= config.excellent_min:
        return "Excellent"
    if score >= config.good_min:
        return "Good"
    if score >= config.average_min:
        return "Average"
    return "Poor"


def _not_assessed(criterion_id: str, name: str, weight: int, explanation: str) -> CriterionResult:
    return CriterionResult(
        id=criterion_id,
        name=name,
        weight=weight,
        rating=None,
        status="not_assessed",
        explanation=explanation,
    )


def _assessed(
    criterion_id: str,
    name: str,
    weight: int,
    rating: int,
    explanation: str,
    timestamps: list[str] | None = None,
) -> CriterionResult:
    return CriterionResult(
        id=criterion_id,
        name=name,
        weight=weight,
        rating=rating,
        status="assessed",
        explanation=explanation,
        timestamps=timestamps or [],
    )


def _opening(features: Features, config: RubricConfig) -> CriterionResult:
    name = "Opening and rapport"
    if not features.transcript_reliable:
        return _not_assessed("opening", name, config.opening_weight, "Opening was not assessed because the transcript is unreliable or missing.")
    greetings = len(features.greeting_mentions)
    students = features.student_turns_in_opening
    if greetings >= 2 and students >= 1:
        rating = 2
    elif greetings >= 1 or students >= 1:
        rating = 1
    else:
        rating = 0
    return _assessed(
        "opening",
        name,
        config.opening_weight,
        rating,
        f"Greeting cues in the first 8 minutes: {greetings}. Student turns in that window: {students}.",
        [item.timestamp for item in features.greeting_mentions],
    )


def _inclusion(features: Features, config: RubricConfig) -> CriterionResult:
    name = "Inclusion and individual engagement"
    share = features.student_turn_share
    if not features.transcript_reliable or share is None or features.distinct_students < 1:
        return _not_assessed(
            "inclusion",
            name,
            config.inclusion_weight,
            "Inclusion was not assessed because student participation could not be measured reliably.",
        )
    if share >= config.inclusion_high:
        rating = 2
    elif share >= config.inclusion_mid:
        rating = 1
    else:
        rating = 0
    if features.distinct_students < 2 and rating == 2:
        rating = 1
    return _assessed(
        "inclusion",
        name,
        config.inclusion_weight,
        rating,
        (
            f"Students spoke {share:.0%} of labelled turns "
            f"across {features.distinct_students} distinct students."
        ),
    )


def _closure(features: Features, config: RubricConfig) -> CriterionResult:
    name = "Lesson structure and closure"
    if not features.transcript_reliable:
        return _not_assessed("closure", name, config.closure_weight, "Closure was not assessed because the transcript is unreliable or missing.")
    rating = 2 if features.closing_mentions else 0
    return _assessed(
        "closure",
        name,
        config.closure_weight,
        rating,
        "A closing phrase was found near the end." if rating else "No closing phrase was found in the last 5 minutes.",
        [item.timestamp for item in features.closing_mentions],
    )


def _homework(features: Features, config: RubricConfig) -> CriterionResult:
    name = "Homework assigned"
    if not features.transcript_reliable:
        return _not_assessed("homework", name, config.homework_weight, "Homework was not assessed because the transcript is unreliable or missing.")
    stamps = [item.timestamp for item in features.homework_mentions]
    rating = 2 if stamps else 0
    explanation = f"Homework is mentioned at {', '.join(stamps)}." if stamps else "No homework mention was found."
    return _assessed("homework", name, config.homework_weight, rating, explanation, stamps)


def _technical(media: MediaStats | None, config: RubricConfig) -> CriterionResult:
    name = "Technical quality"
    if media is None:
        return _not_assessed("technical", name, config.technical_weight, "Technical quality was not assessed because the recording could not be measured.")
    if not media.audio_present:
        return _assessed("technical", name, config.technical_weight, 0, "The recording has no audio stream.")
    if media.loudness_lufs is not None and media.loudness_lufs < SEVERE_LOUDNESS_LUFS:
        return _assessed(
            "technical",
            name,
            config.technical_weight,
            0,
            f"Integrated loudness is {media.loudness_lufs:.1f} LUFS, below {SEVERE_LOUDNESS_LUFS:.0f} LUFS.",
        )
    return _assessed(
        "technical",
        name,
        config.technical_weight,
        2,
        "Audio is present and loudness is in a usable range. Silence and picture brightness were measured but not treated as quality judgments.",
    )


def _aggregate(
    criteria: list[CriterionResult], config: RubricConfig
) -> tuple[float | None, str | None, str | None, list[str]]:
    """Score assessed criteria only. The model does not choose the class."""
    assessed = [item for item in criteria if item.status == "assessed" and item.rating is not None]
    assessed_weight = sum(item.weight for item in assessed)
    if assessed_weight <= 0:
        return None, None, None, ["Insufficient evidence to assign a class."]
    score = 100.0 * sum(item.weight * (item.rating or 0) / 2 for item in assessed) / assessed_weight
    if assessed_weight < config.min_assessed_weight:
        return score, None, None, ["Insufficient evidence to assign a class."]
    classification = classify_score(score, config)
    technical = next((item for item in criteria if item.id == "technical"), None)
    cap = None
    if technical is not None and technical.rating == 0:
        cap = "technical"
        if classification in ("Excellent", "Good"):
            classification = "Average"
    return score, classification, cap, []


def _average_rating(left: int, right: int) -> int:
    """Average two 0–2 ratings. A half rounds up, so 1 and 2 become 2."""
    return (left + right + 1) // 2


_RULES_KEPT = "The judge did not return usable evidence, so this score is the rules score. "


def align_ai_with_rules(rules: Report, ai: Report, config: RubricConfig | None = None) -> Report:
    """Use the rules score where the judge left a scored criterion blank.

    Dropping that criterion makes the remaining perfect scores look like 100.
    Criteria that neither side can score stay open.
    """
    config = config or RubricConfig()
    rules_by = {item.id: item for item in rules.criteria}
    aligned: list[CriterionResult] = []
    filled = False
    for item in ai.criteria:
        rule = rules_by.get(item.id)
        judge_scored = item.status == "assessed" and item.rating is not None
        rules_scored = rule is not None and rule.status == "assessed" and rule.rating is not None
        if not judge_scored and rules_scored:
            aligned.append(replace(rule, explanation=_RULES_KEPT + rule.explanation))
            filled = True
        else:
            aligned.append(item)
    known = {item.id for item in aligned}
    for item in rules.criteria:
        if item.id not in known and item.status == "assessed" and item.rating is not None:
            aligned.append(replace(item, explanation=_RULES_KEPT + item.explanation))
            filled = True
    if not filled:
        return ai
    score, classification, cap, _extra = _aggregate(aligned, config)
    return replace(
        ai,
        classification=classification,
        score=score,
        cap_applied=cap,
        criteria=aligned,
        strengths=_findings(aligned, 2),
        improvements=_findings(aligned, 0),
    )


def limit_ai_credit(rules: Report, ai: Report, config: RubricConfig | None = None) -> Report:
    """Stop the judge from awarding more credit than the measured evidence.

    On the labelled classes, weaker recordings were marked down for camera,
    tone, and background. Those are not judged here. A missing or higher
    judge rating must not turn the remaining full marks into a better class.
    """
    config = config or RubricConfig()
    rules_by = {item.id: item for item in rules.criteria}
    limited: list[CriterionResult] = []
    changed = False
    seen: set[str] = set()
    for item in ai.criteria:
        seen.add(item.id)
        rule = rules_by.get(item.id)
        rules_scored = rule is not None and rule.status == "assessed" and rule.rating is not None
        judge_scored = item.status == "assessed" and item.rating is not None
        if rules_scored and judge_scored and item.rating > rule.rating:
            limited.append(
                replace(
                    rule,
                    explanation=(
                        f"The judge rated this {item.rating} / 2. "
                        f"Measured evidence supports {rule.rating} / 2, so the higher rating was not used. "
                        + rule.explanation
                    ),
                )
            )
            changed = True
        elif rules_scored and not judge_scored:
            limited.append(
                replace(
                    rule,
                    explanation=(
                        "The judge left this blank. "
                        f"Measured evidence supports {rule.rating} / 2, so the missing rating was not treated as full marks. "
                        + rule.explanation
                    ),
                )
            )
            changed = True
        else:
            limited.append(item)
    for item in rules.criteria:
        if item.id in seen or item.status != "assessed" or item.rating is None:
            continue
        limited.append(
            replace(
                item,
                explanation=(
                    "The judge left this blank. "
                    f"Measured evidence supports {item.rating} / 2, so the missing rating was not treated as full marks. "
                    + item.explanation
                ),
            )
        )
        changed = True
    if not changed:
        return ai
    score, classification, cap, _extra = _aggregate(limited, config)
    note = (
        "The AI result was not given more credit than the measured evidence. "
        "Camera, tone, and background are not judged from the transcript."
    )
    notes = [item for item in ai.reliability_notes if item != note]
    notes.append(note)
    return replace(
        ai,
        classification=classification,
        score=score,
        cap_applied=cap,
        criteria=limited,
        strengths=_findings(limited, 2),
        improvements=_findings(limited, 0),
        reliability_notes=notes,
    )


def incorporate_ai_context(rules: Report, ai: Report | None, config: RubricConfig | None = None) -> Report:
    """Combine the rules result with the AI result. The rules are the context the judge received."""
    if ai is None:
        return rules
    config = config or RubricConfig()
    ai_by_id = {item.id: item for item in ai.criteria}
    merged: list[CriterionResult] = []
    averaged: list[str] = []
    filled: list[str] = []
    for item in rules.criteria:
        other = ai_by_id.get(item.id)
        rules_scored = item.status == "assessed" and item.rating is not None
        ai_scored = other is not None and other.status == "assessed" and other.rating is not None
        if rules_scored and ai_scored:
            rating = _average_rating(item.rating, other.rating)
            stamps = list(dict.fromkeys([*item.timestamps, *other.timestamps]))
            if item.rating == other.rating:
                explanation = item.explanation
            else:
                explanation = (
                    f"Combined from the rules ({item.rating} / 2) and the AI judge ({other.rating} / 2). "
                    f"{item.explanation} {other.explanation}"
                )
                averaged.append(item.name)
            merged.append(replace(item, rating=rating, explanation=explanation, timestamps=stamps))
        elif rules_scored:
            merged.append(item)
        elif ai_scored:
            merged.append(replace(other, weight=item.weight))
            filled.append(item.name)
        else:
            merged.append(item)
    known = {item.id for item in merged}
    for item in ai.criteria:
        if item.id not in known and item.status == "assessed" and item.rating is not None:
            merged.append(item)
            filled.append(item.name)
    skip = "Insufficient evidence to assign a class."
    notes = [
        note
        for note in rules.reliability_notes
        if note != skip and not note.startswith("Rules-only mode is provisional")
    ]
    for note in ai.reliability_notes:
        if note not in notes and note != skip and not note.startswith("Hybrid mode"):
            notes.append(note)
    if averaged or filled:
        parts = ["The final class combines the rules result and the AI result."]
        if averaged:
            parts.append("Averaged: " + ", ".join(averaged) + ".")
        if filled:
            parts.append("Taken from the AI judge because the rules had not scored them: " + ", ".join(filled) + ".")
        notes.append(" ".join(parts))
        mode = "final"
    elif any(item.status == "assessed" for item in ai.criteria):
        notes.append("The rules and the AI judge agreed on every scored criterion.")
        mode = "final"
    else:
        notes.append("The AI judge did not return a usable score. The class is the rules result.")
        mode = rules.mode
    score, classification, cap, extra = _aggregate(merged, config)
    for note in extra:
        if note not in notes:
            notes.append(note)
    return replace(
        rules,
        classification=classification,
        score=score,
        provisional=True,
        mode=mode,
        cap_applied=cap,
        reliability_notes=notes,
        criteria=merged,
        strengths=_findings(merged, 2),
        improvements=_findings(merged, 0),
    )


def finalize_report(
    criteria: list[CriterionResult],
    features: Features,
    media: MediaStats | None = None,
    config: RubricConfig | None = None,
    mode: str = "rules_only",
    notes: list[str] | None = None,
) -> Report:
    """Turn criterion ratings into a class. The model does not choose the class."""
    config = config or RubricConfig()
    report_notes = list(notes or [])
    score, classification, cap, extra = _aggregate(criteria, config)
    for note in extra:
        if note not in report_notes:
            report_notes.append(note)
    return Report(
        classification=classification,
        score=score,
        provisional=True,
        mode=mode,
        cap_applied=cap,
        reliability_notes=report_notes,
        criteria=criteria,
        strengths=_findings(criteria, 2),
        improvements=_findings(criteria, 0),
        measurements={
            "duration_s": features.duration_s if media is None else media.duration_s,
            "student_turn_share": features.student_turn_share,
            "teacher_words_per_minute": features.teacher_words_per_minute,
            "homework_mentions": len(features.homework_mentions),
            "loudness_lufs": None if media is None else media.loudness_lufs,
            "silence_s": None if media is None else media.silence_s,
            "longest_silence_s": None if media is None else media.longest_silence_s,
            "mean_luminance": None if media is None else media.mean_luminance,
            "dark_frame_ratio": None if media is None else media.dark_frame_ratio,
            "static_frame_ratio": None if media is None else media.static_frame_ratio,
            "frame_count": 0 if media is None else media.frame_count,
        },
    )


def _findings(criteria: list[CriterionResult], rating: int) -> list[str]:
    matched = [item for item in criteria if item.rating == rating]
    matched.sort(key=lambda item: item.weight, reverse=True)
    return [f"{item.name}: {item.explanation}" for item in matched]


def score_rules(
    features: Features,
    media: MediaStats | None = None,
    config: RubricConfig | None = None,
) -> Report:
    config = config or RubricConfig()
    criteria = [
        _opening(features, config),
        _inclusion(features, config),
        _not_assessed(
            "tone",
            "Tone and energy",
            config.tone_weight,
            "Tone was not assessed. A transcript does not show how the teacher sounded.",
        ),
        _closure(features, config),
        _homework(features, config),
        _not_assessed(
            "teacher_presence",
            "Teacher presence on camera",
            config.presence_weight,
            "Camera presence was not assessed. Frames stay on this computer until faces can be blurred.",
        ),
        _not_assessed(
            "branding",
            "Professional setup",
            config.branding_weight,
            "Setup was not assessed. Frames stay on this computer until faces can be blurred.",
        ),
        _technical(media, config),
    ]
    notes = list(features.reliability_reasons)
    notes.append("Rules-only mode is provisional. Tone, camera presence, and branding are not assessed.")
    return finalize_report(criteria, features, media, config, mode="rules_only", notes=notes)
