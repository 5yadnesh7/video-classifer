"""FFmpeg output parsing and failure handling. No dataset recordings."""

import pytest

from class_quality.media import MediaError, measure_recording, parse_audio_report, parse_frame_report, parse_probe


PROBE = """
{
  "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
  "format": {"duration": "37.900000"}
}
"""

AUDIO = """
[Parsed_ebur128_1 @ 000] t: 0.1 I: -70.0 LUFS LRA: 0.0 LU
[silencedetect @ 000] silence_start: 12.0
[silencedetect @ 000] silence_end: 20.5 | silence_duration: 8.5
[silencedetect @ 000] silence_start: 40.0
[silencedetect @ 000] silence_end: 46.0 | silence_duration: 6.0
[Parsed_ebur128_1 @ 000] Summary:
  Integrated loudness:
    I:         -21.8 LUFS
    Threshold: -31.8 LUFS
  Loudness range:
    LRA:         9.5 LU
    Threshold:   0.0 LUFS
"""

FRAMES = """
[Parsed_metadata_2 @ 000] frame:0    pts:0       pts_time:0
[Parsed_metadata_2 @ 000] lavfi.signalstats.YAVG=10
[Parsed_metadata_2 @ 000] lavfi.signalstats.YDIF=0
[Parsed_metadata_2 @ 000] frame:1    pts:1       pts_time:1
[Parsed_metadata_2 @ 000] lavfi.signalstats.YAVG=126
[Parsed_metadata_2 @ 000] lavfi.signalstats.YDIF=0.2
[Parsed_metadata_2 @ 000] frame:2    pts:2       pts_time:2
[Parsed_metadata_2 @ 000] lavfi.signalstats.YAVG=80
[Parsed_metadata_2 @ 000] lavfi.signalstats.YDIF=6.5
"""


def test_probe_parser_reads_duration_and_which_streams_exist():
    probe = parse_probe(PROBE)
    assert probe.duration_s == pytest.approx(37.9)
    assert probe.video_present is True
    assert probe.audio_present is True


def test_audio_parser_uses_the_summary_loudness_and_silence_durations():
    audio = parse_audio_report(AUDIO)
    assert audio.loudness_lufs == pytest.approx(-21.8)
    assert audio.loudness_range_lu == pytest.approx(9.5)
    assert audio.silence_s == pytest.approx(14.5)
    assert audio.longest_silence_s == pytest.approx(8.5)


def test_frame_parser_reports_luminance_dark_frames_and_static_frames_as_measurements():
    frames = parse_frame_report(FRAMES)
    assert frames.frame_count == 3
    assert frames.mean_luminance == pytest.approx((10 + 126 + 80) / 3)
    assert frames.dark_frame_ratio == pytest.approx(1 / 3)
    assert frames.static_frame_ratio == pytest.approx(1 / 2)


def test_measure_recording_assembles_injected_ffmpeg_output():
    calls: list[list[str]] = []

    def runner(command: list[str]) -> tuple[int, str, str]:
        calls.append(command)
        if command[0] == "ffprobe":
            return 0, PROBE, ""
        if "silencedetect" in " ".join(command):
            return 0, "", AUDIO
        return 0, "", FRAMES

    stats = measure_recording("class.mp4", runner=runner)

    assert stats.duration_s == pytest.approx(37.9)
    assert stats.loudness_lufs == pytest.approx(-21.8)
    assert stats.silence_s == pytest.approx(14.5)
    assert stats.frame_count == 3
    assert all(call[-1] == "class.mp4" or "class.mp4" in call for call in calls)
    assert not any(part.endswith((".vtt", ".wav")) for call in calls for part in call)


def test_ffmpeg_failure_raises_media_error_without_leaking_stderr_bodies_of_cues():
    def runner(command: list[str]) -> tuple[int, str, str]:
        return 1, "", "boom: synthetic failure"

    with pytest.raises(MediaError, match="ffmpeg failed"):
        measure_recording("class.mp4", runner=runner)
