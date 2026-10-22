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


def _name_pattern(token: str) -> re.Pattern[str]:
    """Match a whole name. A one-letter name stays case-sensitive so it does not eat words like \"a\"."""
    flags = 0 if len(token) == 1 else re.IGNORECASE
    return re.compile(rf"\b{re.escape(token)}\b", flags)


def _token_replacements(speakers: list[str], labels: dict[str, str]) -> list[tuple[re.Pattern[str], str]]:
    owners: dict[str, tuple[str, str]] = {}
    ambiguous: set[str] = set()
    for name in speakers:
        for token in name.split():
            if not token:
                continue
            key = token if len(token) == 1 else token.casefold()
            previous = owners.get(key)
            if previous is not None and previous[1] != name:
                ambiguous.add(key)
            else:
                owners[key] = (token, name)
    tokens = [
        (original, labels[owner])
        for key, (original, owner) in owners.items()
        if key not in ambiguous
    ]
    tokens.sort(key=lambda item: (-len(item[0]), item[0].casefold()))
    return [(_name_pattern(token), pseudo) for token, pseudo in tokens]


_PROTECTED_LABEL = re.compile(r"\b(?:TEACHER|UNKNOWN|STUDENT_\d+)\b")


def scrub_names(text: str, names: list[str], replacement: str = "[redacted]") -> str:
    """Replace real names of any length. Longer names are replaced first."""
    protected: dict[str, str] = {}

    def hold(match: re.Match[str]) -> str:
        key = f"\x00{len(protected)}\x00"
        protected[key] = match.group(0)
        return key

    text = _PROTECTED_LABEL.sub(hold, text)
    ordered = sorted(
        {name.strip() for name in names if name and name.strip()},
        key=lambda name: (-len(name), name.casefold()),
    )
    for name in ordered:
        text = _name_pattern(name).sub(replacement, text)
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


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
        (_name_pattern(name), labels[name])
        for name in sorted(labels, key=lambda item: (-len(item), item.casefold()))
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
