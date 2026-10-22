"""VTT parsing. Synthetic cues only; no dataset transcripts."""

from class_quality.vtt import parse_vtt, parse_vtt_document


SAMPLE = """WEBVTT

1
00:01:25.810 --> 00:01:27.000
Alex Example: Hello?

2
00:01:32.160 --> 00:01:33.570
Sam Example: Good evening,
how are you?
"""


def test_parse_vtt_reads_timestamps_speaker_and_joined_text():
    cues = parse_vtt(SAMPLE)

    assert len(cues) == 2
    assert cues[0].start_s == 85.810
    assert cues[0].end_s == 87.000
    assert cues[0].speaker == "Alex Example"
    assert cues[0].text == "Hello?"
    assert cues[1].start_s == 92.160
    assert cues[1].speaker == "Sam Example"
    assert cues[1].text == "Good evening, how are you?"


def test_empty_and_header_only_documents_have_no_cues():
    assert parse_vtt("") == []
    assert parse_vtt("   \n\n  ") == []
    assert parse_vtt("WEBVTT\n\nNOTE this is only a comment\n") == []


def test_malformed_blocks_are_skipped_and_a_valid_cue_is_kept():
    text = """WEBVTT

1
this block has no timestamp

2
00:00:01.000 --> not-a-time
Ada Lovelace: ignored

3
00:00:03.000 --> 00:00:04.500
Ada Lovelace: Hello there
"""
    cues = parse_vtt(text)
    assert len(cues) == 1
    assert cues[0].speaker == "Ada Lovelace"
    assert cues[0].text == "Hello there"
    assert cues[0].start_s == 3.0
    assert cues[0].end_s == 4.5


def test_cue_without_speaker_label_keeps_the_full_text():
    text = """WEBVTT

00:00:10.000 --> 00:00:11.000
just words, no name
"""
    cues = parse_vtt(text)
    assert cues[0].speaker == ""
    assert cues[0].text == "just words, no name"


def test_malformed_blocks_are_reported_as_warnings():
    text = """WEBVTT

1
this block has no timestamp

2
00:00:03.000 --> 00:00:04.000
Ada Lovelace: Hello
"""
    document = parse_vtt_document(text)
    assert [cue.text for cue in document.cues] == ["Hello"]
    assert document.warnings
    assert any("timestamp" in warning.lower() for warning in document.warnings)
