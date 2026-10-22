"""Measurable transcript signals. Synthetic cues only."""

from class_quality.features import extract_features
from class_quality.vtt import Cue


def _cue(start: float, speaker: str, text: str, end: float | None = None) -> Cue:
    return Cue(start_s=start, end_s=start + 2 if end is None else end, speaker=speaker, text=text)


def _lesson(homework: str = "Please do the homework tonight.") -> list[Cue]:
    return [
        _cue(30, "TEACHER", "Hello, how are you?"),
        _cue(50, "STUDENT_1", "I am fine."),
        _cue(70, "TEACHER", "Welcome, please introduce yourself?"),
        _cue(90, "STUDENT_2", "My turn."),
        _cue(400, "TEACHER", "What is the next move?"),
        _cue(490, "STUDENT_1", "This one."),
        _cue(500, "TEACHER", homework),
        _cue(560, "TEACHER", "See you next class, bye."),
    ]


def test_participation_homework_and_timestamps_come_from_the_teacher():
    features = extract_features(_lesson(), duration_s=600)

    assert features.student_turn_share == 0.375
    assert features.distinct_students == 2
    assert features.teacher_question_count == 3
    assert len(features.homework_mentions) == 1
    assert features.homework_mentions[0].timestamp == "00:08:20"
    assert features.homework_mentions[0].timestamp_s == 500

    student_homework = extract_features(
        [_cue(10, "STUDENT_1", "I finished the homework?")],
        duration_s=60,
    )
    assert student_homework.homework_mentions == []


def test_homework_matches_whole_words_only():
    cues = [
        _cue(1, "TEACHER", "Show the board and the answer."),
        _cue(2, "TEACHER", "The HW is on the worksheet?"),
        _cue(3, "TEACHER", "Finish the assignment."),
    ]
    mentions = extract_features(cues, duration_s=30).homework_mentions
    assert [item.timestamp_s for item in mentions] == [2, 3]


def test_opening_window_is_the_first_eight_minutes_and_closing_is_the_last_five():
    cues = _lesson() + [
        _cue(20 * 60, "TEACHER", "Hello again much later."),
    ]
    features = extract_features(cues, duration_s=600)

    assert [item.timestamp_s for item in features.greeting_mentions] == [30, 70]
    assert features.student_turns_in_opening == 2
    assert [item.timestamp_s for item in features.closing_mentions] == [560]
    assert features.teacher_words_per_minute is not None
    assert features.teacher_words_per_minute > 0


def test_non_latin_transcript_is_marked_unreliable():
    cues = [_cue(i, "TEACHER", "hello мир " * 3) for i in range(6)]
    features = extract_features(cues, duration_s=30)

    assert features.non_latin_letter_ratio > 0.15
    assert features.transcript_reliable is False
    assert any("language" in reason.lower() for reason in features.reliability_reasons)


def test_too_few_cues_are_insufficient_evidence():
    features = extract_features(
        [_cue(0, "TEACHER", "Hello?"), _cue(2, "STUDENT_1", "Hi")],
        duration_s=600,
    )

    assert features.transcript_reliable is False
    assert any("few" in reason.lower() or "insufficient" in reason.lower() for reason in features.reliability_reasons)


def test_a_transcript_covering_little_of_the_recording_is_unreliable():
    cues = [_cue(i, "TEACHER", "Hello there friend?") for i in range(6)]
    features = extract_features(cues, duration_s=3600)

    assert features.transcript_reliable is False
    assert any("cover" in reason.lower() for reason in features.reliability_reasons)


def test_missing_transcript_is_reported_as_no_evidence():
    features = extract_features([], duration_s=600)

    assert features.cue_count == 0
    assert features.transcript_reliable is False
    assert features.student_turn_share is None
    assert any("no transcript" in reason.lower() for reason in features.reliability_reasons)
