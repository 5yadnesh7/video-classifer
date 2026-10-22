"""Command line entry point for the rules-only analyzer."""

import subprocess
from pathlib import Path

from class_quality.cli import main
from class_quality.features import Features, Mention
from class_quality.rubric import MediaStats, score_rules


def _report():
    features = Features(
        duration_s=600.0,
        cue_count=12,
        student_turn_share=0.55,
        student_time_share=0.5,
        distinct_students=3,
        teacher_question_count=4,
        teacher_words_per_minute=140.0,
        homework_mentions=[Mention(500.0, "00:08:20")],
        greeting_mentions=[Mention(30.0, "00:00:30"), Mention(70.0, "00:01:10")],
        student_turns_in_opening=2,
        closing_mentions=[Mention(560.0, "00:09:20")],
        non_latin_letter_ratio=0.0,
        transcript_reliable=True,
        reliability_reasons=[],
    )
    media = MediaStats(
        duration_s=600.0,
        audio_present=True,
        video_present=True,
        loudness_lufs=-18.0,
        loudness_range_lu=8.0,
        silence_s=10.0,
        longest_silence_s=4.0,
        mean_luminance=80.0,
        dark_frame_ratio=0.0,
        static_frame_ratio=0.1,
        frame_count=10,
    )
    return score_rules(features, media)


def test_cli_prints_a_provisional_rules_report(capsys, monkeypatch):
    monkeypatch.setattr("class_quality.cli.analyze", lambda *args, **kwargs: _report())

    code = main(["--video", "session.mp4", "--vtt", "session.vtt"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Excellent" in output
    assert "provisional" in output.lower()
    assert "rules_only" in output
    assert "not assessed" in output.lower()
    assert "00:08:20" in output


def test_cli_requires_a_video_path(capsys):
    code = main([])
    assert code != 0
    assert "video" in capsys.readouterr().err.lower()


def test_module_entrypoint_on_a_synthetic_clip(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    vtt = tmp_path / "clip.vtt"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=gray:s=160x90:d=2:r=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    vtt.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nTutor Example: Hello?\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [".venv/Scripts/python.exe", "-m", "class_quality", "--video", str(clip), "--vtt", str(vtt)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Tutor" not in completed.stdout
    assert "Example" not in completed.stdout
    assert "provisional" in completed.stdout.lower()
    assert "Insufficient evidence" in completed.stdout or "not assessed" in completed.stdout.lower()
