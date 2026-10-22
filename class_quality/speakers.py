"""Choose the teacher from question counts and replace real names."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from class_quality.vtt import Cue


@dataclass(frozen=True)
class PseudonymisedTranscript:
    cues: list[Cue]
    teacher: str | None
    students: list[str]


def _named_speakers(cues: list[Cue]) -> list[str]:
    seen: list[str] = []
    for cue in cues:
        if cue.speaker and cue.speaker not in seen:
            seen.append(cue.speaker)
    return seen


def _select_teacher(cues: list[Cue], speakers: list[str]) -> str | None:
    if not speakers:
        return None
    questions: Counter[str] = Counter()
    turns: Counter[str] = Counter()
    first_index: dict[str, int] = {}
    for index, cue in enumerate(cues):
        if cue.speaker not in speakers:
            continue
        questions[cue.speaker] += cue.text.count("?")
        turns[cue.speaker] += 1
        first_index.setdefault(cue.speaker, index)
    return max(
        speakers,
        key=lambda name: (questions[name], turns[name], -first_index[name]),
    )


def _token_replacements(speakers: list[str], labels: dict[str, str]) -> list[tuple[re.Pattern[str], str]]:
    owners: dict[str, str] = {}
    ambiguous: set[str] = set()
    for name in speakers:
        for token in name.split():
            if len(token) < 3:
                continue
            if token in owners and owners[token] != name:
                ambiguous.add(token)
            else:
                owners[token] = name
    tokens = [
        (token, labels[owner])
        for token, owner in owners.items()
        if token not in ambiguous
    ]
    tokens.sort(key=lambda item: len(item[0]), reverse=True)
    return [(re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE), pseudo) for token, pseudo in tokens]


def _redact(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> str:
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text


def pseudonymise(cues: list[Cue]) -> PseudonymisedTranscript:
    speakers = _named_speakers(cues)
    teacher_name = _select_teacher(cues, speakers)
    labels: dict[str, str] = {}
    students: list[str] = []
    if teacher_name is not None:
        labels[teacher_name] = "TEACHER"
        for name in speakers:
            if name == teacher_name:
                continue
            students.append(f"STUDENT_{len(students) + 1}")
            labels[name] = students[-1]

    full_name_patterns = [
        (re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), labels[name])
        for name in sorted(labels, key=len, reverse=True)
    ]
    patterns = full_name_patterns + _token_replacements(speakers, labels)

    redacted = [
        Cue(
            start_s=cue.start_s,
            end_s=cue.end_s,
            speaker=labels.get(cue.speaker, ""),
            text=_redact(cue.text, patterns),
        )
        for cue in cues
    ]
    return PseudonymisedTranscript(
        cues=redacted,
        teacher="TEACHER" if teacher_name else None,
        students=students,
    )
