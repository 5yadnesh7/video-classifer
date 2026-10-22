"""Teacher selection and name pseudonymisation. Synthetic names only."""

import re

from class_quality.speakers import pseudonymise
from class_quality.vtt import Cue


def _cue(start: float, speaker: str, text: str, end: float | None = None) -> Cue:
    return Cue(start_s=start, end_s=end if end is not None else start + 1, speaker=speaker, text=text)


def test_teacher_is_who_asks_the_most_questions_not_who_talks_the_most():
    cues = [
        _cue(0, "Pat Smith", " ".join(["word"] * 40)),
        _cue(10, "Pat Smith", " ".join(["word"] * 40)),
        _cue(20, "Quinn Jones", "What is this piece?"),
        _cue(30, "Quinn Jones", "And this one?"),
        _cue(40, "Quinn Jones", "Can you show me?"),
    ]

    result = pseudonymise(cues)

    teacher_turns = [cue for cue in result.cues if cue.speaker == "TEACHER"]
    assert len(teacher_turns) == 3
    assert all("?" in cue.text for cue in teacher_turns)
    assert result.teacher == "TEACHER"
    assert result.students == ["STUDENT_1"]


def test_question_tie_uses_turn_count_then_who_spoke_first():
    more_turns = [
        _cue(0, "Quiet Person", "Ready?"),
        _cue(5, "Busy Person", "A statement."),
        _cue(10, "Busy Person", "Ready?"),
    ]
    by_turns = pseudonymise(more_turns)
    assert by_turns.cues[1].speaker == "TEACHER"

    same_turns = [
        _cue(0, "Early Speaker", "Ready?"),
        _cue(5, "Later Speaker", "Ready?"),
        _cue(10, "Early Speaker", "A statement."),
        _cue(15, "Later Speaker", "Another statement."),
    ]
    by_order = pseudonymise(same_turns)
    assert by_order.cues[0].speaker == "TEACHER"
    assert by_order.cues[1].speaker == "STUDENT_1"


def test_short_names_of_every_length_are_pseudonymised_without_touching_other_words():
    cues = [
        _cue(0, "Bo Li", "Hello Bo, Also Al went to the algebra class?"),
        _cue(5, "A", "Hi BO and A, a cat sat."),
        _cue(10, "Ann", "Anna Lee saw Ann again."),
        _cue(15, "Anna Lee", "Thanks Ann and Anna Lee."),
    ]

    first = pseudonymise(cues)
    second = pseudonymise(cues)
    rendered = " ".join(f"{cue.speaker}: {cue.text}" for cue in first.cues)

    for secret in ("Bo", "BO", "Li", "Anna Lee", "Ann"):
        assert re.search(rf"\b{re.escape(secret)}\b", rendered) is None
    assert "Also" in rendered
    assert "algebra" in rendered
    assert "a cat" in rendered
    assert "TEACHER" in rendered
    assert "STUDENT_" in rendered
    assert [(cue.speaker, cue.text) for cue in first.cues] == [(cue.speaker, cue.text) for cue in second.cues]


def test_real_names_are_removed_from_text_including_unique_name_parts():
    cues = [
        _cue(0, "Pat Smith", "Hello Quinn Jones"),
        _cue(5, "Quinn Jones", "Hi Pat, please answer?"),
        _cue(10, "Quinn Jones", "Thanks Smith?"),
    ]

    result = pseudonymise(cues)
    rendered = " ".join(f"{cue.speaker}: {cue.text}" for cue in result.cues)

    for secret in ("Pat", "Smith", "Quinn", "Jones"):
        assert secret not in rendered
    assert "TEACHER" in rendered
    assert "STUDENT_1" in rendered


def test_name_redaction_matches_either_case():
    cues = [
        _cue(0, "Pat Smith", "Hello pat, thanks SMITH"),
        _cue(5, "Quinn Jones", "Hi PAT?"),
    ]

    rendered = " ".join(f"{cue.speaker}: {cue.text}" for cue in pseudonymise(cues).cues)

    for secret in ("pat", "PAT", "smith", "SMITH", "quinn", "JONES"):
        assert secret not in rendered


def test_shared_first_names_are_not_guessed_but_full_labels_are_replaced():
    cues = [
        _cue(0, "Sam Lee", "Thanks Sam?"),
        _cue(5, "Sam Roy", "Hello Sam Lee"),
    ]

    result = pseudonymise(cues)
    rendered = " ".join(cue.text for cue in result.cues)

    assert "Sam Lee" not in rendered
    assert "Sam Roy" not in rendered
    assert "Sam" in rendered


def test_a_single_speaker_is_the_teacher_and_there_are_no_students():
    result = pseudonymise([_cue(0, "Only Person", "Hello?")])

    assert result.teacher == "TEACHER"
    assert result.students == []
    assert result.cues[0].speaker == "TEACHER"


def test_unlabelled_cues_stay_unlabelled_and_do_not_become_students():
    cues = [
        _cue(0, "Tutor Name", "Question one?"),
        _cue(2, "", "background noise"),
    ]

    result = pseudonymise(cues)

    assert result.students == []
    assert result.cues[1].speaker == ""
    assert "Tutor" not in result.cues[0].text
