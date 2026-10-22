"""Rules-baseline evaluation. Synthetic files only; labels are for scoring, not prediction."""

import json
from pathlib import Path

from class_quality.evaluation import evaluate_dataset, local_predict, main, render_evaluation
from class_quality.rubric import CriterionResult, Report


def _report(classification: str | None, *criteria: CriterionResult) -> Report:
    return Report(
        classification=classification,
        score=None if classification is None else 50.0,
        provisional=True,
        mode="rules_only",
        cap_applied=None,
        reliability_notes=[],
        criteria=list(criteria),
        strengths=[],
        improvements=["SecretName did well"],
        measurements={},
    )


def _criterion(criterion_id: str, rating: int | None, status: str = "assessed") -> CriterionResult:
    return CriterionResult(
        id=criterion_id,
        name=criterion_id,
        weight=10,
        rating=rating,
        status=status,
        explanation="SecretName was mentioned here",
        timestamps=["00:01:00"],
    )


def _layout(root: Path) -> None:
    files = {
        "excellent": [("alpha session.mp4", b"alpha"), ("beta session.mp4", b"beta")],
        "good": [("gamma session.mp4", b"gamma")],
        "average_": [("delta session.mp4", b"delta")],
        "weak": [("epsilon session.mp4", b"epsilon")],
        "quality issue_": [("zeta session.mp4", b"zeta")],
        "demo_": [("eta session.mp4", b"eta")],
        "notes": [("ignore.txt", b"not a recording")],
    }
    transcripts = {
        "excellent": [("alpha transcript.vtt", "alpha"), ("beta transcript.vtt", "beta")],
        "good": [("gamma transcript.vtt", "gamma")],
        "average_": [("delta transcript.vtt", "delta")],
        "weak": [("epsilon transcript.vtt", "epsilon")],
        "quality issue_": [("zeta transcript.vtt", "zeta")],
        "demo_": [("eta transcript.vtt", "eta")],
    }
    for folder, pairs in files.items():
        directory = root / folder
        directory.mkdir()
        for name, payload in pairs:
            (directory / name).write_bytes(payload)
    for folder, pairs in transcripts.items():
        for name, token in pairs:
            (root / folder / name).write_text(f"WEBVTT\n\n{token}\n", encoding="utf-8")


def test_evaluation_scores_labels_without_giving_them_to_the_classifier(tmp_path: Path):
    _layout(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls: list[tuple[Path, Path | None]] = []

    def predict(video: Path, vtt: Path | None) -> Report:
        calls.append((video, vtt))
        assert "expected" not in predict.__code__.co_varnames
        token = video.read_bytes()
        if token == b"alpha":
            return _report("Good", _criterion("homework", 0), _criterion("tone", None, "not_assessed"))
        if token == b"beta":
            return _report(None, _criterion("opening", None, "not_assessed"))
        if token in {b"epsilon", b"zeta"}:
            return _report("Poor", _criterion("homework", 2))
        if token == b"delta":
            return _report("Poor", _criterion("inclusion", 0))
        return _report("Excellent", _criterion("homework", 2))

    result = evaluate_dataset(tmp_path, predict)
    rendered = render_evaluation(result)

    labelled_videos = [video.name for video, _ in calls]
    assert "eta session.mp4" not in labelled_videos
    assert set(labelled_videos) == {
        "alpha session.mp4",
        "beta session.mp4",
        "gamma session.mp4",
        "delta session.mp4",
        "epsilon session.mp4",
        "zeta session.mp4",
    }
    paired = {video.name: None if vtt is None else vtt.name for video, vtt in calls}
    assert paired["alpha session.mp4"] == "alpha transcript.vtt"
    assert paired["beta session.mp4"] == "beta transcript.vtt"

    assert result.provisional_mapping is True
    assert result.labelled_count == 6
    assert result.unlabelled_excluded == 1
    assert result.correct == 2
    assert result.accuracy == 2 / 6
    assert result.insufficient_evidence == 1
    # Distance 0 or 1 counts. The unrated Excellent recording does not.
    assert result.within_one == 5
    assert result.confusion["Excellent"]["Good"] == 1
    assert result.confusion["Excellent"]["none"] == 1
    assert result.confusion["Good"]["Excellent"] == 1
    assert result.confusion["Average"]["Poor"] == 1
    assert result.confusion["Poor"]["Poor"] == 2
    assert result.per_class["Poor"]["labelled"] == 2
    assert result.per_class["Poor"]["correct"] == 2
    assert result.per_class["Excellent"]["correct"] == 0
    assert result.error_reasons["homework_rated_0"] == 1
    assert result.error_reasons["inclusion_rated_0"] == 1
    assert result.error_reasons["insufficient_evidence"] == 1
    assert "tone_not_assessed" not in result.error_reasons

    assert "SecretName" not in rendered
    assert "alpha" not in rendered.lower()
    assert "session" not in rendered.lower()
    assert "quality issue" not in rendered.lower()
    assert "Excellent" in rendered
    assert "2/6" in rendered
    assert "provisional" in rendered.lower()
    assert {path: path.read_bytes() for path in before} == before
    assert json.loads(result.to_json())["correct"] == 2


def test_local_predict_uses_analyze_and_does_not_receive_a_label(tmp_path: Path, monkeypatch):
    video = tmp_path / "clip.mp4"
    transcript = tmp_path / "clip.vtt"
    video.write_bytes(b"synthetic")
    transcript.write_text("WEBVTT\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_analyze(video_path, vtt_path=None, **kwargs):
        seen["video"] = video_path
        seen["vtt"] = vtt_path
        seen["extra"] = kwargs
        return _report("Average")

    monkeypatch.setattr("class_quality.evaluation.analyze", fake_analyze)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network call")),
    )

    report = local_predict(video, transcript)

    assert report.classification == "Average"
    assert seen["video"] == str(video)
    assert seen["vtt"] == str(transcript)
    assert "label" not in seen["extra"]
    assert "folder" not in seen["extra"]


def test_cli_prints_an_aggregate_and_leaves_the_tree_unchanged(tmp_path: Path, monkeypatch, capsys):
    _layout(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def fake_analyze(video_path, vtt_path=None, **kwargs):
        return _report("Poor", _criterion("homework", 0))

    monkeypatch.setattr("class_quality.evaluation.analyze", fake_analyze)
    code = main(["--dataset", str(tmp_path)])
    output = capsys.readouterr().out

    assert code == 0
    assert "Accuracy:" in output
    assert "SecretName" not in output
    assert "alpha" not in output.lower()
    assert {path: path.read_bytes() for path in before} == before
