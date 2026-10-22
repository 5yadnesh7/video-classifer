"""Command-line rules-only report. No network calls."""

from __future__ import annotations

import argparse
import sys

from class_quality.analyze import analyze
from class_quality.media import sample_frames
from class_quality.rubric import Report
from class_quality.transcribe import transcribe_recording


def render_report(report: Report) -> str:
    score = "n/a" if report.score is None else f"{report.score:.1f}"
    lines = [
        f"Classification: {report.classification or 'not assigned'}",
        f"Score: {score}",
        f"Mode: {report.mode}",
        "This rating is provisional.",
    ]
    if report.cap_applied:
        lines.append(f"Cap applied: {report.cap_applied}")
    lines.append("")
    lines.append("Criteria:")
    for item in report.criteria:
        rating = "not assessed" if item.status != "assessed" else str(item.rating)
        lines.append(f"- {item.name} (weight {item.weight}): {rating}")
        lines.append(f"  {item.explanation}")
        if item.timestamps:
            lines.append("  Timestamps: " + ", ".join(item.timestamps))
    if report.strengths:
        lines.extend(["", "Strengths:"])
        lines.extend(f"- {item}" for item in report.strengths[:3])
    if report.improvements:
        lines.extend(["", "Improvements:"])
        lines.extend(f"- {item}" for item in report.improvements[:3])
    lines.extend(["", "Reliability:"])
    lines.extend(f"- {note}" for note in report.reliability_notes)
    lines.extend(["", "Measurements:"])
    for key, value in report.measurements.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="class_quality", description="Rules-only class quality report.")
    parser.add_argument("--video", required=True, help="Path to the class recording (mp4).")
    parser.add_argument("--vtt", default=None, help="Optional Zoom WebVTT transcript.")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code or 1)
    report = analyze(
        args.video,
        vtt_path=args.vtt,
        transcribe=None if args.vtt else transcribe_recording,
        sample_frames=sample_frames,
    )
    sys.stdout.write(render_report(report))
    return 0
