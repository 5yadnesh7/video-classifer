"""Transcript signals that can be measured without a model."""

from __future__ import annotations

import re
from dataclasses import dataclass

from class_quality.vtt import Cue

OPENING_WINDOW_S = 8 * 60
CLOSING_WINDOW_S = 5 * 60
MIN_CUES = 5
MAX_NON_LATIN_RATIO = 0.15
MIN_COVERAGE = 0.20

_HOMEWORK = re.compile(r"\b(?:homework|home\s+work|hw|assignment)\b", re.IGNORECASE)
_GREETING = re.compile(
    r"\b(?:hello|hi|welcome|introduce|good\s+(?:morning|evening|afternoon)|how\s+are\s+you)\b",
    re.IGNORECASE,
)
_CLOSING = re.compile(
    r"\b(?:bye|goodbye|see\s+you|next\s+class|thank\s+you|good\s+night)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Mention:
    timestamp_s: float
    timestamp: str


@dataclass(frozen=True)
class Features:
    duration_s: float | None
    cue_count: int
    student_turn_share: float | None
    student_time_share: float | None
    distinct_students: int
    teacher_question_count: int
    teacher_words_per_minute: float | None
    homework_mentions: list[Mention]
    greeting_mentions: list[Mention]
    student_turns_in_opening: int
    closing_mentions: list[Mention]
    non_latin_letter_ratio: float
    transcript_reliable: bool
    reliability_reasons: list[str]


def format_timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _mention(cue: Cue) -> Mention:
    return Mention(timestamp_s=cue.start_s, timestamp=format_timestamp(cue.start_s))


def _is_teacher(cue: Cue) -> bool:
    return cue.speaker == "TEACHER"


def _is_student(cue: Cue) -> bool:
    return cue.speaker.startswith("STUDENT_")


def _share(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _non_latin_ratio(cues: list[Cue]) -> float:
    letters = 0
    non_latin = 0
    for cue in cues:
        for char in cue.text:
            if char.isalpha():
                letters += 1
                if ord(char) > 127:
                    non_latin += 1
    if letters == 0:
        return 0.0
    return non_latin / letters


def _reliability(cues: list[Cue], duration_s: float | None, non_latin: float) -> list[str]:
    if not cues:
        return ["No transcript was supplied."]
    reasons: list[str] = []
    if len(cues) < MIN_CUES:
        reasons.append("Too few transcript cues for a reliable reading.")
    if non_latin > MAX_NON_LATIN_RATIO:
        reasons.append(
            "Transcript language does not look like English, so conversational signals are unreliable."
        )
    if duration_s is not None and duration_s >= 60 and cues:
        span = max(cue.end_s for cue in cues) - min(cue.start_s for cue in cues)
        if span / duration_s < MIN_COVERAGE:
            reasons.append("Transcript covers too little of the recording.")
    return reasons


def extract_features(cues: list[Cue], duration_s: float | None = None) -> Features:
    labelled = [cue for cue in cues if _is_teacher(cue) or _is_student(cue)]
    teacher = [cue for cue in cues if _is_teacher(cue)]
    students = [cue for cue in cues if _is_student(cue)]
    labelled_time = sum(max(0.0, cue.end_s - cue.start_s) for cue in labelled)
    student_time = sum(max(0.0, cue.end_s - cue.start_s) for cue in students)
    teacher_time = sum(max(0.0, cue.end_s - cue.start_s) for cue in teacher)
    teacher_words = sum(len(cue.text.split()) for cue in teacher)
    end_s = duration_s if duration_s is not None else (max((cue.end_s for cue in cues), default=0.0))
    closing_from = end_s - CLOSING_WINDOW_S
    non_latin = _non_latin_ratio(cues)
    reasons = _reliability(cues, duration_s, non_latin)

    return Features(
        duration_s=duration_s if duration_s is not None else (end_s or None),
        cue_count=len(cues),
        student_turn_share=_share(len(students), len(labelled)),
        student_time_share=_share(student_time, labelled_time),
        distinct_students=len({cue.speaker for cue in students}),
        teacher_question_count=sum(cue.text.count("?") for cue in teacher),
        teacher_words_per_minute=(teacher_words / (teacher_time / 60) if teacher_time > 0 else None),
        homework_mentions=[_mention(cue) for cue in teacher if _HOMEWORK.search(cue.text)],
        greeting_mentions=[
            _mention(cue)
            for cue in teacher
            if cue.start_s < OPENING_WINDOW_S and _GREETING.search(cue.text)
        ],
        student_turns_in_opening=sum(1 for cue in students if cue.start_s < OPENING_WINDOW_S),
        closing_mentions=[
            _mention(cue) for cue in teacher if cue.start_s >= closing_from and _CLOSING.search(cue.text)
        ],
        non_latin_letter_ratio=non_latin,
        transcript_reliable=not reasons,
        reliability_reasons=reasons,
    )
