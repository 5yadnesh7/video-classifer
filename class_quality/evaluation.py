"""Score an existing rules-only classifier against folder labels.

Folder names are used only to score predictions. They are not given to the
classifier, and this module does not change the rubric.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from class_quality.analyze import analyze
from class_quality.rubric import Report

CLASS_ORDER = ("Poor", "Average", "Good", "Excellent")
FOLDER_LABELS = {
    "excellent": "Excellent",
    "good": "Good",
    "average_": "Average",
    "weak": "Poor",
    "quality issue_": "Poor",
}
UNLABELLED_FOLDERS = {"demo_"}


def _tokens(path: Path) -> set[str]:
    return {token.lower() for token in re.split(r"[^A-Za-z0-9]+", path.stem) if len(token) >= 3}


def _pair(videos: list[Path], transcripts: list[Path]) -> list[tuple[Path, Path | None]]:
    remaining = set(transcripts)
    pairs: list[tuple[Path, Path | None]] = []
    for video in sorted(videos, key=lambda path: path.name):
        ranked = sorted(remaining, key=lambda transcript: len(_tokens(video) & _tokens(transcript)), reverse=True)
        choice = None
        if ranked and _tokens(video) & _tokens(ranked[0]):
            choice = ranked[0]
        elif len(videos) == 1 and len(ranked) == 1:
            choice = ranked[0]
        if choice is not None:
            remaining.remove(choice)
        pairs.append((video, choice))
    return pairs


@dataclass
class EvaluationResult:
    provisional_mapping: bool
    labelled_count: int
    unlabelled_excluded: int
    correct: int
    accuracy: float
    within_one: int
    insufficient_evidence: int
    confusion: dict[str, dict[str, int]]
    per_class: dict[str, dict[str, int]]
    error_reasons: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "provisional_mapping": self.provisional_mapping,
                "labelled_count": self.labelled_count,
                "unlabelled_excluded": self.unlabelled_excluded,
                "correct": self.correct,
                "accuracy": self.accuracy,
                "within_one": self.within_one,
                "insufficient_evidence": self.insufficient_evidence,
                "confusion": self.confusion,
                "per_class": self.per_class,
                "error_reasons": self.error_reasons,
            },
            indent=2,
            sort_keys=True,
        )


def _blank_confusion() -> dict[str, dict[str, int]]:
    columns = [*CLASS_ORDER, "none"]
    return {label: {column: 0 for column in columns} for label in CLASS_ORDER}


def _within_one(expected: str, predicted: str | None) -> bool:
    if predicted not in CLASS_ORDER:
        return False
    return abs(CLASS_ORDER.index(expected) - CLASS_ORDER.index(predicted)) <= 1


def _error_reasons(report: Report) -> Counter[str]:
    reasons: Counter[str] = Counter()
    if report.classification is None:
        reasons["insufficient_evidence"] += 1
        return reasons
    for criterion in report.criteria:
        if criterion.status == "assessed" and criterion.rating == 0:
            reasons[f"{criterion.id}_rated_0"] += 1
    return reasons


def evaluate_dataset(root: Path, predict) -> EvaluationResult:
    """Run `predict(video, transcript)` and compare with folder labels."""
    confusion = _blank_confusion()
    per_class = {label: {"labelled": 0, "correct": 0} for label in CLASS_ORDER}
    reasons: Counter[str] = Counter()
    labelled = 0
    correct = 0
    within_one = 0
    insufficient = 0
    unlabelled = 0

    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        videos = list(directory.glob("*.mp4"))
        if directory.name in UNLABELLED_FOLDERS:
            unlabelled += len(videos)
            continue
        expected = FOLDER_LABELS.get(directory.name)
        if expected is None or not videos:
            continue
        for video, transcript in _pair(videos, list(directory.glob("*.vtt"))):
            report = predict(video, transcript)
            predicted = report.classification if report.classification in CLASS_ORDER else None
            labelled += 1
            per_class[expected]["labelled"] += 1
            confusion[expected]["none" if predicted is None else predicted] += 1
            if predicted is None:
                insufficient += 1
            if predicted == expected:
                correct += 1
                per_class[expected]["correct"] += 1
            else:
                reasons.update(_error_reasons(report))
            if _within_one(expected, predicted):
                within_one += 1

    return EvaluationResult(
        provisional_mapping=True,
        labelled_count=labelled,
        unlabelled_excluded=unlabelled,
        correct=correct,
        accuracy=(correct / labelled) if labelled else 0.0,
        within_one=within_one,
        insufficient_evidence=insufficient,
        confusion=confusion,
        per_class=per_class,
        error_reasons=dict(reasons),
    )


def local_predict(video: Path, transcript: Path | None) -> Report:
    """Run the rules-only analyzer. The label is intentionally not an argument."""
    return analyze(str(video), None if transcript is None else str(transcript))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the rules-only classifier on a local dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset root. Folder names are used only for scoring.")
    args = parser.parse_args(argv)
    result = evaluate_dataset(Path(args.dataset), local_predict)
    sys.stdout.write(render_evaluation(result))
    return 0


def render_evaluation(result: EvaluationResult) -> str:
    """Aggregate text only. No paths, names, or free-text explanations."""
    percent = 100.0 * result.accuracy
    within_percent = 100.0 * result.within_one / result.labelled_count if result.labelled_count else 0.0
    lines = [
        "Rules-only baseline evaluation",
        "Label mapping is provisional.",
        f"Labelled recordings: {result.labelled_count}",
        f"Unlabelled recordings excluded: {result.unlabelled_excluded}",
        f"Accuracy: {result.correct}/{result.labelled_count} ({percent:.1f}%)",
        f"Within one class: {result.within_one}/{result.labelled_count} ({within_percent:.1f}%)",
        f"Insufficient evidence: {result.insufficient_evidence}",
        "",
        "Confusion matrix (row is the reference class, column is the prediction):",
    ]
    header = " ".join(f"{column:>10}" for column in [*CLASS_ORDER, "none"])
    lines.append(f"{'':<12}{header}")
    for label in CLASS_ORDER:
        counts = " ".join(f"{result.confusion[label][column]:>10}" for column in [*CLASS_ORDER, "none"])
        lines.append(f"{label:<12}{counts}")
    lines.extend(["", "Per class:"])
    for label in CLASS_ORDER:
        stats = result.per_class[label]
        lines.append(f"- {label}: {stats['correct']}/{stats['labelled']} correct")
    lines.extend(["", "Reasons counted on incorrect or missing predictions:"])
    if not result.error_reasons:
        lines.append("- none")
    for reason, count in sorted(result.error_reasons.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {reason}: {count}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
