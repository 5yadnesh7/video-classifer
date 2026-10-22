"""Zoom WebVTT parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TIMESTAMP = re.compile(
    r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})"
)
_RANGE = re.compile(
    r"(?P<start>\d+:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d+:\d{2}:\d{2}\.\d{3})"
)


@dataclass(frozen=True)
class Cue:
    start_s: float
    end_s: float
    speaker: str
    text: str


@dataclass(frozen=True)
class VttDocument:
    cues: list[Cue]
    warnings: list[str]


def _seconds(stamp: str) -> float:
    match = _TIMESTAMP.fullmatch(stamp.strip())
    if match is None:
        raise ValueError(f"unrecognised timestamp: {stamp}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int(match.group("ms"))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def parse_vtt(content: str) -> list[Cue]:
    """Parse a WebVTT document into cues. Cues without a time range are skipped."""
    return parse_vtt_document(content).cues


def parse_vtt_document(content: str) -> VttDocument:
    """Parse cues and report blocks that look like cues but have no usable time range."""
    cues: list[Cue] = []
    warnings: list[str] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0] == "WEBVTT" or lines[0].startswith("NOTE"):
            continue
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        match = _RANGE.search(lines[time_index]) if time_index is not None else None
        if match is None:
            warnings.append("Skipped a block with no usable timestamp.")
            continue
        body = " ".join(lines[time_index + 1 :]).strip()
        speaker = ""
        text = body
        if ": " in body:
            label, remainder = body.split(": ", 1)
            if label and len(label) <= 80 and "-->" not in label:
                speaker = label.strip()
                text = remainder.strip()
        cues.append(
            Cue(
                start_s=_seconds(match.group("start")),
                end_s=_seconds(match.group("end")),
                speaker=speaker,
                text=text,
            )
        )
    return VttDocument(cues=cues, warnings=warnings)
