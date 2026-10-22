"""Rules-only scoring. No filenames, identities, or dataset labels."""

from dataclasses import replace

import pytest

from class_quality.features import Features, Mention
from class_quality.rubric import (
    CriterionResult,
    MediaStats,
    Report,
    RubricConfig,
    classify_score,
    incorporate_ai_context,
    score_rules,
)


def _features(**overrides: object) -> Features:
    values: dict[str, object] = dict(
        duration_s=600.0,
        cue_count=20,
        student_turn_share=0.55,
        student_time_share=0.50,
        distinct_students=4,
        teacher_question_count=10,
        teacher_words_per_minute=150.0,
        homework_mentions=[Mention(500.0, "00:08:20")],
        greeting_mentions=[Mention(30.0, "00:00:30"), Mention(70.0, "00:01:10")],
        student_turns_in_opening=3,
        closing_mentions=[Mention(560.0, "00:09:20")],
        non_latin_letter_ratio=0.0,
        transcript_reliable=True,
        reliability_reasons=[],
    )
    values.update(overrides)
    return Features(**values)  # type: ignore[arg-type]


def _media(**overrides: object) -> MediaStats:
    values: dict[str, object] = dict(
        duration_s=600.0,
        audio_present=True,
        video_present=True,
        loudness_lufs=-18.0,
        loudness_range_lu=10.0,
        silence_s=30.0,
        longest_silence_s=12.0,
        mean_luminance=80.0,
        dark_frame_ratio=0.0,
        static_frame_ratio=0.1,
        frame_count=20,
    )
    values.update(overrides)
    return MediaStats(**values)  # type: ignore[arg-type]


def _by_id(report, criterion_id: str):
    return next(item for item in report.criteria if item.id == criterion_id)


def test_grade_boundaries():
    assert classify_score(80) == "Excellent"
    assert classify_score(79.9) == "Good"
    assert classify_score(65) == "Good"
    assert classify_score(64.9) == "Average"
    assert classify_score(50) == "Average"
    assert classify_score(49.9) == "Poor"


def test_strong_measurable_signals_score_excellent_and_leave_ai_criteria_unassessed():
    report = score_rules(_features(), _media(dark_frame_ratio=0.95, static_frame_ratio=0.9))

    assert report.mode == "rules_only"
    assert report.provisional is True
    assert report.classification == "Excellent"
    assert report.score == 100
    assert report.cap_applied is None
    for criterion_id in ("tone", "teacher_presence", "branding"):
        item = _by_id(report, criterion_id)
        assert item.status == "not_assessed"
        assert item.rating is None
    assert "visible" not in _by_id(report, "teacher_presence").explanation.lower()
    assert "not assessed" in _by_id(report, "tone").explanation.lower()
    assert "not assessed" in _by_id(report, "branding").explanation.lower()
    assert "00:08:20" in _by_id(report, "homework").explanation
    assert _by_id(report, "technical").rating == 2


def test_weak_signals_score_poor_without_using_speaking_pace():
    weak = dict(
        student_turn_share=0.10,
        distinct_students=2,
        greeting_mentions=[],
        student_turns_in_opening=0,
        closing_mentions=[],
        homework_mentions=[],
    )
    slow = score_rules(_features(teacher_words_per_minute=40.0, **weak), _media())
    fast = score_rules(_features(teacher_words_per_minute=220.0, **weak), _media())

    assert slow.classification == "Poor"
    assert slow.score == fast.score
    assert _by_id(slow, "opening").rating == 0
    assert _by_id(slow, "inclusion").rating == 0
    assert _by_id(slow, "closure").rating == 0
    assert _by_id(slow, "homework").rating == 0


def test_one_opening_signal_is_a_partial_rating():
    report = score_rules(
        _features(greeting_mentions=[Mention(30.0, "00:00:30")], student_turns_in_opening=0),
        _media(),
    )
    assert _by_id(report, "opening").rating == 1


def test_inclusion_thresholds_and_single_student_cap():
    high = score_rules(_features(student_turn_share=0.55, distinct_students=2), _media())
    mid = score_rules(_features(student_turn_share=0.45, distinct_students=2), _media())
    low = score_rules(_features(student_turn_share=0.20, distinct_students=2), _media())
    one_student = score_rules(_features(student_turn_share=0.70, distinct_students=1), _media())

    assert _by_id(high, "inclusion").rating == 2
    assert _by_id(mid, "inclusion").rating == 1
    assert _by_id(low, "inclusion").rating == 0
    assert _by_id(one_student, "inclusion").rating == 1


def test_long_silence_and_a_dark_picture_do_not_cap_or_lower_the_class():
    report = score_rules(
        _features(),
        _media(silence_s=2000, longest_silence_s=400, dark_frame_ratio=0.95, mean_luminance=5),
    )

    assert report.classification == "Excellent"
    assert report.cap_applied is None
    assert _by_id(report, "technical").rating == 2
    assert _by_id(report, "teacher_presence").status == "not_assessed"


def test_inaudible_audio_caps_the_class_at_average_and_does_not_raise_a_poor_score():
    capped = score_rules(_features(), _media(loudness_lufs=-40))
    assert _by_id(capped, "technical").rating == 0
    assert capped.score > 80
    assert capped.classification == "Average"
    assert capped.cap_applied == "technical"

    already_poor = score_rules(
        _features(
            greeting_mentions=[],
            student_turns_in_opening=0,
            student_turn_share=0.1,
            closing_mentions=[],
            homework_mentions=[],
        ),
        _media(audio_present=False, loudness_lufs=None),
    )
    assert already_poor.classification == "Poor"
    assert already_poor.cap_applied == "technical"


def test_unreliable_transcript_is_not_scored_as_zero():
    report = score_rules(
        _features(
            transcript_reliable=False,
            reliability_reasons=["Transcript language does not look like English."],
            homework_mentions=[],
            greeting_mentions=[],
        ),
        _media(),
    )

    for criterion_id in ("opening", "inclusion", "closure", "homework", "tone"):
        assert _by_id(report, criterion_id).status == "not_assessed"
        assert _by_id(report, criterion_id).rating is None
    assert report.classification is None
    assert any("insufficient" in note.lower() for note in report.reliability_notes)


def test_no_transcript_and_no_media_is_insufficient_evidence():
    report = score_rules(
        _features(
            cue_count=0,
            student_turn_share=None,
            transcript_reliable=False,
            reliability_reasons=["No transcript was supplied."],
            greeting_mentions=[],
            closing_mentions=[],
            homework_mentions=[],
            student_turns_in_opening=0,
            distinct_students=0,
        ),
        None,
    )

    assert report.classification is None
    assert report.score is None
    assert all(item.status == "not_assessed" for item in report.criteria)
    assert report.provisional is True


def test_boundaries_are_configurable_and_strengths_are_the_highest_clear_criteria():
    report = score_rules(_features(), _media(), RubricConfig(excellent_min=101))

    assert report.score == 100
    assert report.classification == "Good"
    assert report.strengths
    assert "Homework" in report.strengths[0] or any("Homework" in item for item in report.strengths)
    joined = " ".join(report.strengths + report.improvements)
    assert "Pat" not in joined


def _criterion(criterion_id, name, weight, rating, status, explanation):
    return CriterionResult(criterion_id, name, weight, rating, status, explanation, [])


def _bare_report(criteria, mode="rules_only"):
    return Report(
        classification=None,
        score=None,
        provisional=True,
        mode=mode,
        cap_applied=None,
        reliability_notes=["Rules-only mode is provisional. Tone, camera presence, and branding are not assessed."],
        criteria=criteria,
        strengths=[],
        improvements=[],
        measurements={"duration_s": 600.0, "loudness_lufs": -18.0},
    )


def test_a_blank_judge_score_cannot_become_an_excellent_total():
    from class_quality.rubric import align_ai_with_rules

    rules = _bare_report(
        [
            _criterion("opening", "Opening and rapport", 15, 2, "assessed", "Rules opening."),
            _criterion("inclusion", "Inclusion and individual engagement", 20, 0, "assessed", "Students spoke 30%."),
            _criterion("closure", "Lesson structure and closure", 10, 2, "assessed", "Rules closure."),
            _criterion("homework", "Homework assigned", 10, 2, "assessed", "Rules homework."),
            _criterion("technical", "Technical quality", 10, 2, "assessed", "Audio is present."),
            _criterion("tone", "Tone and energy", 15, None, "not_assessed", "Tone was not assessed."),
        ]
    )
    ai = replace(
        _bare_report(
            [
                _criterion("opening", "Opening and rapport", 15, 2, "assessed", "AI opening."),
                _criterion("inclusion", "Inclusion and individual engagement", 20, None, "not_assessed", "No usable evidence."),
                _criterion("closure", "Lesson structure and closure", 10, 2, "assessed", "AI closure."),
                _criterion("homework", "Homework assigned", 10, 2, "assessed", "AI homework."),
                _criterion("technical", "Technical quality", 10, 2, "assessed", "AI technical."),
                _criterion("tone", "Tone and energy", 15, None, "not_assessed", "Tone was not assessed."),
            ],
            mode="hybrid",
        ),
        classification="Excellent",
        score=100.0,
    )

    aligned = align_ai_with_rules(rules, ai)
    inclusion = next(item for item in aligned.criteria if item.id == "inclusion")

    assert inclusion.rating == 0
    assert inclusion.status == "assessed"
    assert aligned.classification == "Good"
    assert aligned.score == pytest.approx(100 * 45 / 65)
    assert next(item for item in aligned.criteria if item.id == "tone").status == "not_assessed"


def test_ai_credit_cannot_rise_above_the_measured_rating():
    from class_quality.rubric import limit_ai_credit

    rules = _bare_report(
        [
            _criterion("opening", "Opening and rapport", 15, 1, "assessed", "One greeting."),
            _criterion("inclusion", "Inclusion and individual engagement", 20, 0, "assessed", "Student share is 0.30."),
            _criterion("closure", "Lesson structure and closure", 10, 2, "assessed", "The class closes."),
            _criterion("homework", "Homework assigned", 10, 2, "assessed", "Homework is mentioned."),
            _criterion("technical", "Technical quality", 10, 2, "assessed", "Audio is present."),
            _criterion("tone", "Tone and energy", 15, None, "not_assessed", "Tone was not assessed."),
            _criterion("teacher_presence", "Teacher presence on camera", 15, None, "not_assessed", "Camera was not judged."),
            _criterion("branding", "Professional setup", 5, None, "not_assessed", "Background was not judged."),
        ]
    )
    rules = replace(rules, classification="Poor", score=42.3)
    ai = replace(
        _bare_report(
            [
                _criterion("opening", "Opening and rapport", 15, 2, "assessed", "Warm opening."),
                _criterion("inclusion", "Inclusion and individual engagement", 20, 2, "assessed", "Students joined in."),
                _criterion("closure", "Lesson structure and closure", 10, 2, "assessed", "Clear close."),
                _criterion("homework", "Homework assigned", 10, 0, "assessed", "No homework found."),
                _criterion("technical", "Technical quality", 10, 2, "assessed", "Audio is clear."),
                _criterion("tone", "Tone and energy", 15, None, "not_assessed", "Tone was not assessed."),
                _criterion("teacher_presence", "Teacher presence on camera", 15, None, "not_assessed", "Camera was not judged."),
                _criterion("branding", "Professional setup", 5, None, "not_assessed", "Background was not judged."),
            ],
            mode="hybrid",
        ),
        classification="Excellent",
        score=100.0,
    )

    limited = limit_ai_credit(rules, ai)
    by_id = {item.id: item for item in limited.criteria}

    assert by_id["opening"].rating == 1
    assert by_id["inclusion"].rating == 0
    assert by_id["homework"].rating == 0
    assert "higher rating was not used" in by_id["inclusion"].explanation
    assert limited.classification == "Poor"
    assert limited.score == pytest.approx(100 * 27.5 / 65)
    assert limited.score < 50
    assert "not given more credit" in " ".join(limited.reliability_notes)


def test_final_averages_shared_scores_and_keeps_a_score_only_one_side_could_make():
    rules = _bare_report(
        [
            _criterion("opening", "Opening and rapport", 15, 2, "assessed", "Rules opening."),
            _criterion("inclusion", "Inclusion and individual engagement", 20, None, "not_assessed", "No student labels."),
            _criterion("technical", "Technical quality", 10, 2, "assessed", "Audio is present."),
        ]
    )
    ai = _bare_report(
        [
            _criterion("opening", "Opening and rapport", 15, 0, "assessed", "AI opening."),
            _criterion("inclusion", "Inclusion and individual engagement", 20, 1, "assessed", "AI inclusion."),
            _criterion("technical", "Technical quality", 10, 0, "assessed", "AI technical."),
        ],
        mode="hybrid",
    )

    final = incorporate_ai_context(rules, ai)
    by_id = {item.id: item for item in final.criteria}

    assert by_id["opening"].rating == 1
    assert "combined" in by_id["opening"].explanation.lower()
    assert by_id["inclusion"].rating == 1
    assert by_id["inclusion"].explanation == "AI inclusion."
    assert by_id["technical"].rating == 1
    assert final.score == pytest.approx(50.0)
    assert final.classification == "Average"
    assert final.mode == "final"
    assert "Opening and rapport" in " ".join(final.reliability_notes)
    assert final.measurements["duration_s"] == 600.0


def test_ai_evidence_does_not_assign_a_class_below_the_weight_floor():
    rules = _bare_report(
        [_criterion("technical", "Technical quality", 10, 2, "assessed", "Audio is present.")]
    )
    rules = Report(
        classification=None,
        score=100.0,
        provisional=True,
        mode="rules_only",
        cap_applied=None,
        reliability_notes=["Insufficient evidence to assign a class."],
        criteria=rules.criteria,
        strengths=[],
        improvements=[],
        measurements={"duration_s": 30.0},
    )
    ai = _bare_report(
        [_criterion("branding", "Professional setup", 5, 2, "assessed", "AI branding.")],
        mode="hybrid",
    )

    final = incorporate_ai_context(rules, ai)

    assert final.classification is None
    assert final.score == pytest.approx(100.0)
    assert any(item.id == "branding" and item.rating == 2 for item in final.criteria)


def test_missing_ai_context_leaves_the_rules_report_unchanged():
    rules = score_rules(_features(), _media())

    assert incorporate_ai_context(rules, None) is rules
