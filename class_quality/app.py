"""Local Gradio interface for a provisional class-quality report.

Rules-only analysis is the default. Ollama Cloud stays off until the project
.env file contains both values after the organiser has approved cloud
processing:

    OLLAMA_CLOUD_PERMITTED=1
    OLLAMA_API_KEY=<key from https://ollama.com>

The application reads that file on startup. A variable already set in the
shell is left unchanged.

Any other permission value, including ``true``, leaves cloud analysis off.
This module never sets either variable and never turns ``permitted`` on unless
``OLLAMA_CLOUD_PERMITTED`` is already exactly ``1``.

Run it from the project directory:

    .venv\\Scripts\\python.exe -m class_quality.app

The page binds to 127.0.0.1. Recordings are copied to neutral temporary names
and those copies are deleted when the analysis finishes.

Uploads larger than 8 GiB are rejected from their on-disk size before that
copy. 8 GiB covers a 90-minute class at about 12 Mbps (about 7.5 GiB), which
is the high end of a 40–90 minute local recording, with a little headroom.
"""

from __future__ import annotations

import base64
import html
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import gradio as gr

from class_quality.analyze import analyze, attach_transcript_metadata
from class_quality.cli import render_report
from class_quality.env import load_local_env
from class_quality.features import extract_features
from class_quality.judge import EvidencePack, FrameStill, RubricJudge, cloud_enabled
from class_quality.media import MediaError, measure_recording
from class_quality.media import sample_frames as sample_local_frames
from class_quality.ollama_client import OllamaCloudClient
from class_quality.rubric import MediaStats, Report, limit_ai_credit, score_rules
from class_quality.speakers import pseudonymise, scrub_names
from class_quality.transcribe import failed_transcript, transcribe_recording
from class_quality.vtt import parse_vtt

_INTRO = """# Class quality

Upload an MP4 recording and, if you have one, a WebVTT transcript.

The result is provisional. Tone and rapport are not judged from transcript text. Teacher visibility and branded background stay not assessed when no still images are available.

The AI judge is unavailable until cloud processing is permitted by the organiser. After that approval, set `OLLAMA_CLOUD_PERMITTED=1` and `OLLAMA_API_KEY` in the project `.env` file, then start this page again. Any other value, including `true`, keeps the judge off. This application does not turn permission on by itself.

Recordings larger than 8 GiB are rejected before they are copied. That ceiling fits a 40–90 minute class, including a 90-minute recording at about 12 Mbps.
"""

# 90 minutes at about 12 Mbps is roughly 7.5 GiB. 8 GiB leaves headroom for a
# 40–90 minute class without accepting an unbounded local copy.
MAX_UPLOAD_BYTES = 8 * 1024**3


_CLOUD_OFF = """
The AI judge is unavailable until cloud processing is permitted.
This is the provisional rules-only result. Tone, rapport, teacher visibility, and branded background are not judged from the transcript.
After the organiser approves Ollama Cloud, set OLLAMA_CLOUD_PERMITTED=1 and OLLAMA_API_KEY in the project .env file, then run the interface again.
"""


def validate_upload(
    video_path: str | None,
    vtt_path: str | None = None,
    max_bytes: int | None = None,
) -> str | None:
    """Return a user-facing error, or None when the upload can be analysed."""
    limit = MAX_UPLOAD_BYTES if max_bytes is None else max_bytes
    if not video_path:
        return "Upload an MP4 recording before analysis can start."
    video = _as_path(video_path)
    if video.suffix.lower() != ".mp4":
        return "Only MP4 recordings are accepted. Upload a file that ends in .mp4."
    if not video.is_file():
        return "The MP4 recording could not be read. Upload the file again."
    if video.stat().st_size == 0:
        return "The recording is empty, so it was not copied. Upload an MP4 that contains the class."
    oversized = _oversized(video, limit, "recording")
    if oversized:
        return oversized
    if vtt_path:
        transcript = _as_path(vtt_path)
        if transcript.suffix.lower() != ".vtt":
            return "Optional transcripts must be WebVTT files ending in .vtt."
        if not transcript.is_file():
            return "The transcript file could not be read. Remove it or upload the .vtt again."
        if transcript.stat().st_size == 0:
            return "The transcript is empty, so it was not copied. Remove it or upload a .vtt that contains the class."
        oversized = _oversized(transcript, limit, "transcript")
        if oversized:
            return oversized
    return None


@dataclass
class _Outcome:
    text: str
    report: Report | None = None
    rules: Report | None = None
    ai: Report | None = None


def analyse_upload(
    video_path,
    vtt_path=None,
    progress=gr.Progress(),
    *,
    client=None,
    measure=measure_recording,
    workspace_factory=tempfile.mkdtemp,
    stills_fn=None,
    max_bytes: int | None = None,
    transcribe=None,
    sample_frames=None,
    use_ai=None,
) -> str:
    """Copy the upload to neutral temporary files, score it, then delete the copies."""
    return _analyse(
        video_path,
        vtt_path,
        progress,
        client=client,
        measure=measure,
        workspace_factory=workspace_factory,
        stills_fn=stills_fn,
        max_bytes=max_bytes,
        transcribe=transcribe,
        sample_frames=sample_frames,
        use_ai=use_ai,
    ).text


def _analyse(
    video_path,
    vtt_path=None,
    progress=None,
    *,
    client=None,
    measure=measure_recording,
    workspace_factory=tempfile.mkdtemp,
    stills_fn=None,
    max_bytes: int | None = None,
    transcribe=None,
    sample_frames=None,
    use_ai=None,
    on_step=None,
) -> _Outcome:
    error = validate_upload(video_path, vtt_path, max_bytes=max_bytes)
    if error:
        return _Outcome(error)
    video = _as_path(video_path)
    transcript = _as_path(vtt_path) if vtt_path else None
    names: list[str] = []
    workspace: Path | None = None
    transcribe = transcribe_recording if transcribe is None else transcribe
    sampler = sample_local_frames if sample_frames is None and stills_fn is None else sample_frames
    try:
        names = _speaker_names(transcript)
        workspace = Path(workspace_factory())
        workspace.mkdir(parents=True, exist_ok=True)
        _mark(progress, on_step, 1, 0.15, "Uploading the recording")
        stored_video = workspace / "upload.mp4"
        shutil.copyfile(video, stored_video)
        stored_vtt = None
        local = None
        if transcript is not None:
            stored_vtt = workspace / "upload.vtt"
            shutil.copyfile(transcript, stored_vtt)
        else:
            _mark(progress, on_step, 2, 0.36, "Transcribing the recording locally")
            try:
                local = transcribe(str(stored_video))
            except Exception:
                local = failed_transcript()
            (workspace / "transcript.vtt").write_text(local.vtt_text or "WEBVTT\n\n", encoding="utf-8")
        requested = cloud_enabled() if use_ai is None else bool(use_ai)
        if transcript is not None:
            _mark(progress, on_step, 2, 0.36, "Extracting evidence")
        if requested and cloud_enabled():
            _mark(progress, on_step, 3, 0.66, "Evaluating with the AI judge")
            rules_report, ai_report = _hybrid_report(
                stored_video,
                stored_vtt,
                names,
                measure,
                client,
                stills_fn,
                workspace,
                local,
                sampler,
            )
            ai_report = limit_ai_credit(rules_report, ai_report)
            text = _render_views(rules_report, ai_report)
        else:
            _mark(progress, on_step, 3, 0.66, "Evaluating the class")
            rules_report = analyze(
                str(stored_video),
                None if stored_vtt is None else str(stored_vtt),
                measure=measure,
                local_transcript=local,
                sample_frames=sampler,
            )
            ai_report = None
            if requested and not cloud_enabled():
                blocked = (
                    "The AI switch was on, but this process is not permitted to call Ollama Cloud. "
                    "The judge was not called."
                )
                rules_report = replace(rules_report, reliability_notes=[blocked, *rules_report.reliability_notes])
            text = _render_views(rules_report, None)
            if not cloud_enabled():
                text += _CLOUD_OFF
        _mark(progress, on_step, 4, 1.0, "Report ready")
        return _Outcome(
            _scrub(text, names),
            _scrub_report(rules_report, names),
            _scrub_report(rules_report, names),
            None if ai_report is None else _scrub_report(ai_report, names),
        )
    except Exception as error:
        message = _scrub(f"Analysis failed: {error}", names)
        _mark(progress, on_step, 4, 1.0, "Analysis failed")
        return _Outcome(message)
    finally:
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)


def format_dashboard(report: Report) -> dict[str, str]:
    """Turn a real report into dashboard HTML. Missing evidence stays missing."""
    assessed = [item for item in report.criteria if item.status == "assessed" and item.rating is not None]
    mode = "AI-Assisted Assessment" if report.mode == "hybrid" else "Provisional — Rules Only"
    band = _band_class(report.classification)
    label = html.escape(report.classification or "Class not assigned")
    if report.score is None:
        score = ""
    elif report.classification is None:
        score = (
            f'<p class="cq-score">{report.score:.1f} / 100</p>'
            '<p class="cq-note">This number covers only the criteria that could be scored. It is not a class grade.</p>'
        )
    else:
        score = f'<p class="cq-score">{report.score:.1f} / 100</p>'
    why = _assignment_reason(report)
    cap = ""
    if report.cap_applied:
        cap = f'<p class="cq-note">A cap was applied: {html.escape(report.cap_applied)}.</p>'
    rating = (
        '<section class="cq-result-grid">'
        f'<article class="cq-rating {band}"><p class="cq-kicker">{mode}</p>'
        f"<h2>{label}</h2>{score}{cap}<p>{html.escape(why)}</p></article>"
        f"{_key_info_html(report, len(assessed))}"
        f"{_notes_html(report)}"
        "</section>"
    )
    cards = ['<section class="cq-rubric"><h2>Evaluation criteria</h2><div class="cq-criterion-grid">']
    for item in report.criteria:
        state, badge = _criterion_badge(item.status, item.rating)
        stamps = ""
        if item.timestamps:
            stamps = '<p class="cq-time">Evidence: ' + html.escape(", ".join(item.timestamps)) + "</p>"
        cards.append(
            f'<article class="cq-criterion {state}"><header><h3>{html.escape(item.name)}</h3>'
            f"<span>{badge}</span></header><p>{html.escape(item.explanation)}</p>{stamps}</article>"
        )
    cards.append("</div></section>")
    return {
        "rating": rating,
        "rubric": "".join(cards),
        "findings": _findings_html(report.strengths, report.improvements),
        "metrics": _metrics_html(report.measurements),
    }


def _render_views(rules: Report, ai: Report | None) -> str:
    """Text form of the rules result and the AI result. The rules are the judge's context."""
    parts = ["Rules result:", render_report(rules)]
    if ai is None:
        parts.append(
            "AI result:\nThe AI judge was not used. No transcript evidence was sent to Ollama Cloud."
        )
    else:
        if not _judge_owned(ai):
            parts.append(
                "AI result:\nNot scored.\n"
                "The judge replied, but none of its ratings matched the recording. "
                "Those ratings were not turned into a second class."
            )
        elif ai.classification is None:
            parts.append(
                "AI result:\nNot enough evidence.\n"
                "The rules result was given to the judge as context. "
                "The judge did not score enough criteria to assign a class. "
                "The partial total is not shown as a grade."
            )
        else:
            parts.append(
                "AI result:\nThe rules result was given to the judge as context.\n" + render_report(ai)
            )
    return "\n\n".join(parts)


def _score_line(report: Report) -> str:
    if report.score is None:
        return ""
    if report.classification is None:
        return (
            f'<p class="cq-score">{report.score:.1f} / 100</p>'
            '<p class="cq-note">This number covers only the criteria that could be scored. It is not a class grade.</p>'
        )
    return f'<p class="cq-score">{report.score:.1f} / 100</p>'


def _summary_card(kicker: str, report: Report, note: str, extra_class: str = "") -> str:
    band = _band_class(report.classification)
    label = html.escape(report.classification or "Class not assigned")
    return (
        f'<article class="cq-rating {band} {extra_class}"><p class="cq-kicker">{html.escape(kicker)}</p>'
        f"<h2>{label}</h2>{_score_line(report)}<p>{html.escape(note)}</p></article>"
    )


_RULES_KEPT_PREFIX = "The judge did not return usable evidence, so this score is the rules score."


def _rules_kept(item) -> bool:
    return item.explanation.startswith(_RULES_KEPT_PREFIX)


def _judge_owned(ai: Report) -> list:
    return [
        item
        for item in ai.criteria
        if item.status == "assessed" and item.rating is not None and not _rules_kept(item)
    ]


def _ai_card(rules: Report, ai: Report) -> str:
    """Show the judge's own result. A rules-only fill is not a second class."""
    owned = _judge_owned(ai)
    if not owned:
        return (
            '<article class="cq-rating cq-band-open"><p class="cq-kicker">AI result</p>'
            "<h2>Not scored</h2>"
            "<p>The judge replied, but none of its ratings matched the recording. "
            "Those ratings were not turned into a second class.</p></article>"
        )
    if ai.classification is None:
        return (
            '<article class="cq-rating cq-band-open"><p class="cq-kicker">AI result</p>'
            "<h2>Not enough evidence</h2>"
            f"<p>The rules result was given to the judge as context. The judge scored {len(owned)} criteria. "
            "That is not enough to assign a class, so the partial total is not shown as a grade.</p></article>"
        )
    filled = sum(1 for item in ai.criteria if _rules_kept(item))
    limited = any(
        item.explanation.startswith("The judge rated this ") or item.explanation.startswith("The judge left this blank.")
        for item in ai.criteria
    )
    same = (
        ai.classification == rules.classification
        and ai.score is not None
        and rules.score is not None
        and abs(ai.score - rules.score) < 0.05
    )
    note = "The rules result was given to the judge as context. "
    if limited:
        note = (
            "The AI result was not given more credit than the measured evidence. "
            "Camera, tone, and background are not judged from the transcript."
        )
    elif filled and same:
        note += f"{filled} criteria the judge left blank keep the rules score, so this total matches the rules result."
    elif filled:
        note += f"{filled} criteria the judge left blank keep the rules score."
    elif same:
        note += f"The judge scored {len(owned)} criteria from the evidence, and this total matches the rules result."
    else:
        note += f"The judge scored {len(owned)} criteria from the evidence."
    return _summary_card("AI result", ai, note)


def format_comparison(rules: Report, ai: Report | None) -> dict[str, str]:
    """Show the rules result and the AI result. The rules are the context given to the judge."""
    rules_note = "Scored on this computer from the transcript and recording measurements."
    if ai is None:
        ai_card = (
            '<article class="cq-rating cq-band-open"><p class="cq-kicker">AI result</p>'
            "<h2>Not used</h2>"
            "<p>The AI judge was not used. No transcript evidence was sent to Ollama Cloud.</p></article>"
        )
        mode_label = "Rules only"
        privacy = "This run stayed on this computer. Transcript evidence was not sent to Ollama Cloud."
    else:
        ai_card = _ai_card(rules, ai)
        mode_label = "AI, with the rules as context"
        privacy = (
            "This run sent the rules result and transcript evidence to Ollama Cloud. "
            "The recording and unblurred frames were not sent."
        )
    assessed_count = sum(1 for item in rules.criteria if item.status == "assessed" and item.rating is not None)
    rating = (
        '<section class="cq-compare">'
        + _summary_card("Rules result", rules, rules_note)
        + ai_card
        + "</section>"
        + '<section class="cq-detail">'
        + _key_info_html(rules, assessed_count, mode_label)
        + _notes_html(rules, privacy, judge_ran=ai is not None)
        + "</section>"
    )
    return {
        "rating": rating,
        "rubric": _comparison_rubric(rules, ai),
        "findings": _findings_html(rules.strengths, rules.improvements),
        "metrics": _metrics_html(rules.measurements),
    }


def _comparison_rubric(rules: Report, ai: Report | None) -> str:
    ai_by = {item.id: item for item in ai.criteria} if ai is not None else {}
    cards = ['<section class="cq-rubric"><h2>Evaluation criteria</h2><div class="cq-criterion-grid">']
    for item in rules.criteria:
        ai_item = ai_by.get(item.id)
        rules_kept = ai_item is not None and _rules_kept(ai_item)
        ai_text = _view_badge(ai_item, "Not used")
        if ai_item is not None and ai_item.explanation.startswith("The judge rated this ") and ai_item.rating is not None:
            ai_text = f"{ai_item.rating} / 2, higher judge rating was not used"
        elif ai_item is not None and ai_item.explanation.startswith("The judge left this blank.") and ai_item.rating is not None:
            ai_text = f"{ai_item.rating} / 2, missing rating was not treated as full marks"
        elif rules_kept and ai_item is not None and ai_item.rating is not None:
            ai_text = f"{ai_item.rating} / 2, rules score kept"
        state, _badge = _criterion_badge(item.status, item.rating)
        stamps = ""
        if item.timestamps:
            stamps = '<p class="cq-time">Evidence: ' + html.escape(", ".join(item.timestamps)) + "</p>"
        ai_detail = ""
        if (
            ai_item is not None
            and ai_item.status == "assessed"
            and not rules_kept
            and ai_item.explanation
            and ai_item.explanation != item.explanation
        ):
            ai_detail = f"<p>{html.escape(ai_item.explanation)}</p>"
        cards.append(
            f'<article class="cq-criterion {state}"><header><h3>{html.escape(item.name)}</h3></header>'
            f'<p class="cq-source">Rules: {html.escape(_view_badge(item, "Not used"))}</p>'
            f'<p class="cq-source">AI: {html.escape(ai_text)}</p>'
            f"<p>{html.escape(_shown_explanation(item))}</p>{ai_detail}{_open_stat(item, rules)}{stamps}</article>"
        )
    cards.append("</div></section>")
    return "".join(cards)


def _open_stat(item, report: Report) -> str:
    """Show the local measurement on a card that has no quality score."""
    if item.status == "assessed":
        return ""
    if item.id == "tone":
        rate = report.measurements.get("teacher_words_per_minute")
        if isinstance(rate, (int, float)):
            return f'<p class="cq-source">Measured speaking rate: {float(rate):.0f} words/min. This is not a tone score.</p>'
    if item.id in {"teacher_presence", "branding"}:
        frames = report.measurements.get("frame_count")
        if isinstance(frames, int) and frames > 0:
            return f'<p class="cq-source">Measured frames: {frames}. These frames are not a camera or setup score.</p>'
    return ""


def _shown_explanation(item) -> str:
    """Open criteria say why they stay open, including after the judge has run."""
    if item.status == "assessed":
        return item.explanation
    return {
        "tone": "Tone was not assessed. A transcript does not show how the teacher sounded.",
        "teacher_presence": "Camera presence was not assessed. Frames stay on this computer until faces can be blurred.",
        "branding": "Setup was not assessed. Frames stay on this computer until faces can be blurred.",
    }.get(item.id, item.explanation)


def _view_badge(item, missing: str) -> str:
    if item is None:
        return missing
    _, badge = _criterion_badge(item.status, item.rating)
    return badge


def _switch_info() -> str:
    if cloud_enabled():
        return "Off keeps this run on this computer. On sends transcript evidence to Ollama Cloud. The recording and frames stay here."
    return "This process is not permitted to call Ollama Cloud, so the switch cannot send a transcript."


def _band_class(classification: str | None) -> str:
    return {
        "Poor": "cq-band-poor",
        "Average": "cq-band-average",
        "Good": "cq-band-good",
        "Excellent": "cq-band-excellent",
    }.get(classification or "", "cq-band-open")


def _assignment_reason(report: Report) -> str:
    if report.classification is None:
        return "There is not enough assessed evidence to assign a class. Criteria that could not be assessed were not scored as zero."
    if report.provisional:
        return "This class is provisional. It uses only the criteria that could be assessed."
    return "This class was calculated from the assessed criteria."


def _key_info_html(report: Report, assessed_count: int, mode_label: str | None = None) -> str:
    if mode_label is None:
        mode_label = {"hybrid": "AI-assisted", "final": "Rules, with AI evidence"}.get(report.mode, "Rules only")
    rows = [("Mode", mode_label)]
    if report.provisional:
        rows.append(("Status", "Provisional"))
    duration = report.measurements.get("duration_s")
    if isinstance(duration, (int, float)):
        rows.append(("Recording duration", _format_duration(float(duration))))
    source = report.measurements.get("transcript_source")
    source_label = {"local": "Transcribed locally", "supplied": "Uploaded transcript", "missing": "No transcript"}.get(
        str(source), ""
    )
    if source_label:
        rows.append(("Transcript source", source_label))
    language = report.measurements.get("detected_language")
    if isinstance(language, str) and language:
        probability = report.measurements.get("language_probability")
        shown = {"en": "English"}.get(language.lower(), language)
        if isinstance(probability, (int, float)):
            shown = f"{shown} ({float(probability):.2f})"
        rows.append(("Language detected", shown))
    rows.append(("Criteria assessed", f"{assessed_count} of {len(report.criteria)}"))
    body = "".join(f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>" for label, value in rows)
    return f'<article class="cq-panel"><h2>Key information</h2><dl class="cq-facts">{body}</dl></article>'


def _notes_html(report: Report, privacy: str | None = None, judge_ran: bool = False) -> str:
    if privacy is None:
        if report.mode == "hybrid":
            privacy = "This run sent transcript evidence to Ollama Cloud. The recording and unblurred frames were not sent."
        elif report.mode == "final":
            privacy = (
                "The final class was calculated by the rules. Transcript evidence was sent to Ollama Cloud. "
                "The recording and unblurred frames were not sent."
            )
        else:
            privacy = "This run stayed on this computer. Transcript evidence was not sent to Ollama Cloud."
    items = []
    for item in (privacy, *report.reliability_notes):
        if not item:
            continue
        if judge_ran and (
            item.startswith("Rules-only mode is provisional") or item.startswith("Hybrid mode is provisional")
        ):
            continue
        items.append(item)
    body = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f'<article class="cq-panel"><h2>Important notes</h2><ul>{body}</ul></article>'


def present_analysis(
    video_path,
    vtt_path=None,
    progress=gr.Progress(),
    *,
    client=None,
    measure=measure_recording,
    workspace_factory=tempfile.mkdtemp,
    stills_fn=None,
    max_bytes: int | None = None,
    use_ai=None,
) -> tuple[str, bool, bool, str, str, str, str]:
    """Return an error, empty-state visibility, and dashboard HTML from one real run."""
    outcome = _analyse(
        video_path,
        vtt_path,
        progress,
        client=client,
        measure=measure,
        workspace_factory=workspace_factory,
        stills_fn=stills_fn,
        max_bytes=max_bytes,
        use_ai=use_ai,
    )
    if outcome.report is None:
        return outcome.text, True, False, "", "", "", ""
    view = format_comparison(outcome.rules or outcome.report, outcome.ai)
    return "", False, True, view["rating"], view["rubric"], view["findings"], view["metrics"]


def build_interface() -> gr.Blocks:
    """Build the upload page. Calling this does not start the server or call Ollama."""
    with gr.Blocks(title="ClassQuality AI", analytics_enabled=False, fill_width=True) as demo:
        gr.HTML(_page_copy())
        with gr.Row(elem_classes=["cq-hero"]):
            with gr.Column(scale=6):
                gr.HTML(_intro_copy())
            with gr.Column(scale=5, elem_classes=["cq-upload-card"]):
                with gr.Row(elem_classes=["cq-files"]):
                    video = gr.File(
                        label="Class recording (MP4)",
                        file_types=[".mp4"],
                        type="filepath",
                        elem_classes=["cq-drop"],
                    )
                    transcript = gr.File(
                        label="Optional transcript (VTT)",
                        file_types=[".vtt"],
                        type="filepath",
                        elem_classes=["cq-drop"],
                    )
                ai_switch = gr.Checkbox(
                    label="Use AI judge",
                    value=False,
                    info=_switch_info(),
                    elem_classes=["cq-switch"],
                )
                analyze = gr.Button("Analyze Class", variant="primary", elem_classes=["cq-analyze"])
                gr.HTML(
                    '<p class="cq-upload-note">Supported format: MP4, up to 8 GiB, plus an optional .vtt transcript. '
                    "A 40–90 minute recording can take several minutes.</p>"
                )
        stages = gr.HTML(_stages_html())
        outputs = [*_output_slots(), stages]
        gr.HTML(_privacy_details())
        analyze.click(
            _ui_event,
            inputs=[video, transcript, ai_switch],
            outputs=outputs,
            show_progress="hidden",
        )
    return demo


def _bind_address() -> tuple[str, int]:
    """Local runs stay on localhost. A host sets CLASS_QUALITY_HOST and PORT."""
    host = os.environ.get("CLASS_QUALITY_HOST") or "127.0.0.1"
    raw_port = os.environ.get("PORT") or "7860"
    try:
        port = int(raw_port)
    except ValueError:
        port = 7860
    if port < 1 or port > 65535:
        port = 7860
    return host, port


def main() -> None:
    """Serve the interface on localhost. Cloud analysis stays off unless the environment already allows it.

    ``share=False`` is explicit so GRADIO_SHARE cannot publish the page.
    The host and port change only when the deployment sets CLASS_QUALITY_HOST and PORT.
    """
    load_local_env()
    host, port = _bind_address()
    build_interface().launch(
        server_name=host,
        server_port=port,
        share=False,
        max_file_size=MAX_UPLOAD_BYTES,
        theme=_theme(),
        css=APP_CSS,
        run_history=False,
        footer_links=[],
    )


def _ui_event(video_path, vtt_path=None, use_ai=False, progress=gr.Progress()):
    """Update the stage tracker while the analysis runs, then show one result."""
    events: queue.Queue = queue.Queue()
    holder: list = []

    def on_step(stage: int, fraction: float, message: str) -> None:
        events.put(("step", stage, fraction, message))

    def worker() -> None:
        try:
            holder.append(
                _analyse(
                    video_path,
                    vtt_path,
                    progress=None,
                    use_ai=bool(use_ai),
                    on_step=on_step,
                )
            )
        except Exception as error:
            holder.append(error)
        finally:
            events.put(("done",))

    threading.Thread(target=worker, daemon=True).start()
    stage, fraction, message = 1, 0.08, "Uploading the recording"
    yield _running_view(stage, fraction, message)
    while True:
        try:
            item = events.get(timeout=0.45)
        except queue.Empty:
            cap = {1: 0.28, 2: 0.58, 3: 0.92}.get(stage, 1.0)
            if fraction < cap - 0.004:
                fraction = round(min(cap, fraction + (cap - fraction) * 0.18), 3)
                yield _running_view(stage, fraction, message)
            continue
        if item[0] == "done":
            break
        if item[1] >= 4:
            continue
        _, stage, fraction, message = item
        yield _running_view(stage, fraction, message)
    outcome = holder[0] if holder else _Outcome("Analysis failed.")
    if isinstance(outcome, Exception):
        outcome = _Outcome(f"Analysis failed: {outcome}")
    if outcome.report is None:
        yield (
            _error_html(outcome.text),
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            "",
            "",
            "",
            _stages_html(),
        )
        return
    view = format_comparison(outcome.rules or outcome.report, outcome.ai)
    yield (
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        view["rating"],
        view["rubric"],
        view["findings"],
        view["metrics"],
        _stages_html(5, 1, "Report ready"),
    )


def _running_view(stage: int, fraction: float, message: str):
    return (
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        "",
        "",
        "",
        _stages_html(stage, fraction, message),
    )


def _output_slots():
    error = gr.HTML()
    with gr.Column(visible=True) as empty:
        gr.HTML(_EMPTY)
    with gr.Column(visible=False) as results:
        rating = gr.HTML()
        rubric = gr.HTML()
        with gr.Row(elem_classes=["cq-bottom"]):
            metrics = gr.HTML()
            findings = gr.HTML()
    return [error, empty, results, rating, rubric, findings, metrics]


def _page_copy() -> str:
    if cloud_enabled():
        judge = "AI judge available"
        privacy = "Starts off. Turn the switch on to use the judge."
        privacy_class = "cq-pill cq-pill-warn"
        judge_class = "cq-pill cq-pill-warn"
    else:
        judge = "AI judge unavailable"
        privacy = "Processing stays on this computer."
        privacy_class = "cq-pill cq-pill-ok"
        judge_class = "cq-pill cq-pill-off"
    return f"""
    <header class="cq-header">
      <div class="cq-brand-lockup">
        <span class="cq-mark" aria-hidden="true">CQ</span>
        <div>
          <p class="cq-brand">ClassQuality AI</p>
          <p class="cq-subtitle">Evidence-based evaluation of online teaching quality.</p>
        </div>
      </div>
      <div class="cq-pills">
        <p class="{judge_class}">{html.escape(judge)}</p>
        <p class="{privacy_class}">{html.escape(privacy)}</p>
      </div>
    </header>
    """


def _intro_copy() -> str:
    return """
    <section class="cq-intro">
      <h1>Understand what makes a class effective.</h1>
      <p>Upload a class recording and get an evidence-based assessment of teaching quality using research-backed criteria.</p>
      <div class="cq-categories">
        <article><h2>Poor</h2><p>Below 50. Needs significant improvement.</p></article>
        <article><h2>Average</h2><p>50 to 64. Meets basic expectations.</p></article>
        <article><h2>Good</h2><p>65 to 79. Effective teaching.</p></article>
        <article><h2>Excellent</h2><p>80 and above. Highly effective teaching.</p></article>
      </div>
    </section>
    """


def _stages_html(active: int = 0, fraction: float = 0, message: str = "") -> str:
    """Numbered steps. Earlier steps are complete, the current step is highlighted."""
    steps = (
        (1, "Upload", "MP4 recording and optional transcript"),
        (2, "Extract evidence", "Audio, transcript, and key frames"),
        (3, "Evaluate", "Score only what the evidence supports"),
        (4, "Report", "Class, criteria, and measurements"),
    )
    items = []
    for number, title, detail in steps:
        if active > 4 or (active and number < active):
            klass = "cq-stage-done"
        elif number == active:
            klass = "cq-stage-active"
        else:
            klass = ""
        items.append(
            f'<li class="{klass}"><span>{number}</span><strong>{html.escape(title)}</strong>'
            f"<small>{html.escape(detail)}</small></li>"
        )
    bar = ""
    if active:
        percent = max(0, min(100, round(fraction * 100)))
        bar = (
            f'<div class="cq-progress" role="progressbar" aria-valuenow="{percent}" aria-valuemin="0" aria-valuemax="100">'
            f'<span style="width:{percent}%"></span></div>'
            f'<p class="cq-progress-label">{html.escape(message)} · {percent}%</p>'
        )
    return f'<div class="cq-stage-wrap">{bar}<ol class="cq-stages">{"".join(items)}</ol></div>'


def _privacy_details() -> str:
    local = (
        "Cloud processing is disabled. Rules-only analysis runs on this computer."
        if not cloud_enabled()
        else "Cloud processing is enabled for this process. A run that uses the AI judge sends transcript evidence to Ollama Cloud. Set OLLAMA_CLOUD_PERMITTED only after the organiser approves it."
    )
    return f"""
    <details class="cq-limits">
      <summary>Privacy and limitations</summary>
      <ul>
        <li>{html.escape(local)} The value true does not enable cloud analysis. OLLAMA_CLOUD_PERMITTED must be exactly 1.</li>
        <li>Accepted files: one MP4 recording, up to 8 GiB, and an optional .vtt transcript. A 40–90 minute class can take several minutes.</li>
        <li>If no transcript is attached, speech is transcribed on this computer. A supplied .vtt file is used instead.</li>
        <li>The page shows two results: the rules result, and the AI result. The rules result is given to the judge as context.</li>
        <li>The AI judge is used only when a run is labelled as using AI evidence. Configuring access does not by itself make a result AI-assisted.</li>
        <li>Evidence that could not be assessed is not treated as a failure and is not scored as zero.</li>
        <li>Teacher presence and branded background are not judged from luminance. Unblurred frames are not sent.</li>
        <li>Class recordings are confidential. Keep them on this machine and do not publish this page.</li>
      </ul>
    </details>
    """


_EMPTY = """
<section class="cq-empty">
  <h2>Ready when you are.</h2>
  <p>Add a class recording to see a provisional result. The stages above are not complete until an analysis finishes. No sample class is shown here.</p>
</section>
"""


APP_CSS = """
html, body, .gradio-container, .dark, .dark body, .dark .gradio-container {
  color-scheme: light !important;
  background: #eef2f6 !important;
  color: #142033 !important;
}
.gradio-container, .gradio-container .app, .gradio-container .main, .gradio-container .contain {
  background: #eef2f6 !important;
  max-width: none !important;
  width: 100% !important;
  padding-left: 20px !important;
  padding-right: 20px !important;
}
footer, footer .settings, footer .built-with { display: none !important; }
.cq-drop, .cq-drop label, .cq-drop .label-wrap, .cq-drop span {
  background: #f8fafc !important;
  color: #142033 !important;
  border-color: #cbd5e1 !important;
}
.prose, .prose p, .prose li, .prose h1, .prose h2, .prose h3, .prose dt, .prose dd, .prose strong, .prose span, .prose small {
  opacity: 1 !important;
}
.cq-header, .cq-intro, .cq-empty, .cq-rating, .cq-panel, .cq-criterion, .cq-strengths, .cq-improvements, .cq-metrics, .cq-limits, .cq-upload-card {
  background: #ffffff !important;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  color: #142033 !important;
  box-shadow: 0 1px 2px rgba(16, 32, 51, 0.04);
}
.cq-header h1, .cq-header h2, .cq-header p, .cq-intro h1, .cq-intro h2, .cq-intro p,
.cq-empty h2, .cq-empty p, .cq-rating h2, .cq-rating p, .cq-panel h2, .cq-panel li, .cq-panel dt, .cq-panel dd,
.cq-criterion h3, .cq-criterion p, .cq-strengths h2, .cq-strengths li, .cq-strengths p,
.cq-improvements h2, .cq-improvements li, .cq-improvements p, .cq-metrics h2, .cq-metrics p, .cq-limits li, .cq-stages strong, .cq-stages small {
  color: #142033 !important;
}
.cq-header { display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 16px 20px; margin-bottom: 14px; }
.cq-brand-lockup { display: flex; align-items: center; gap: 12px; }
.cq-mark { width: 36px; height: 36px; border-radius: 10px; background: #0f766e; color: #ffffff !important; display: grid; place-items: center; font-size: 0.78rem; font-weight: 750; }
.cq-brand { margin: 0; font-size: 1.35rem; font-weight: 750; letter-spacing: -0.03em; }
.cq-subtitle { margin: 2px 0 0; color: #526277 !important; font-size: 0.92rem; }
.cq-pills { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.cq-pill { margin: 0; padding: 6px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
.cq-pill-off { background: #fff1f2 !important; color: #9f1239 !important; }
.cq-pill-ok { background: #ecfdf5 !important; color: #047857 !important; }
.cq-pill-warn { background: #fff7ed !important; color: #9a3412 !important; }
.cq-intro { padding: 22px; }
.cq-intro h1 { margin: 0 0 8px; font-size: 1.7rem; letter-spacing: -0.03em; }
.cq-intro > p, .cq-empty p, .cq-note, .cq-time, .cq-criterion p, .cq-panel li, .cq-strengths li, .cq-improvements li, .cq-metrics > p {
  color: #24344a !important; font-size: 0.95rem; line-height: 1.45;
}
.cq-categories { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 16px; }
.cq-categories article { background: #f8fafc !important; border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px; }
.cq-categories h2 { margin: 0 0 4px; font-size: 0.92rem; }
.cq-categories p { margin: 0; color: #3d4d63 !important; font-size: 0.78rem !important; line-height: 1.35; }
.cq-upload-card { padding: 14px; }
.cq-files { gap: 10px !important; }
.cq-upload-note { margin: 8px 2px 0; color: #526277 !important; font-size: 0.8rem; line-height: 1.4; }
.cq-stage-wrap { margin: 16px 0; }
.cq-progress { height: 8px; background: #e2e8f0; border-radius: 999px; overflow: hidden; margin: 0 0 8px; }
.cq-progress span { display: block; height: 100%; background: #0f766e; border-radius: inherit; transition: width 0.35s linear; }
.cq-progress-label { margin: 0 0 10px; color: #334155 !important; font-size: 0.85rem; font-weight: 650; }
.cq-stages { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0; padding: 0; list-style: none; }
.cq-stages li { display: flex; flex-direction: column; gap: 4px; background: #ffffff !important; border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px; }
.cq-stage-active { border-color: #0f766e !important; box-shadow: 0 0 0 2px #99f6e4; }
.cq-stage-done { border-color: #99f6e4 !important; background: #f0fdfa !important; }
.cq-stages span { width: 22px; height: 22px; border-radius: 999px; background: #e6f4f3 !important; color: #0f766e !important; font-size: 0.75rem; font-weight: 750; display: grid; place-items: center; }
.cq-stage-active span, .cq-stage-done span { background: #0f766e !important; color: #ffffff !important; }
.cq-stages small { color: #526277 !important; font-size: 0.78rem; }
.cq-result-grid, .cq-compare { display: grid; grid-template-columns: 1fr 1fr 1.05fr; gap: 16px; margin-top: 8px; }
.cq-compare { grid-template-columns: 1fr 1fr; margin-bottom: 16px; }
.cq-detail { display: grid !important; grid-template-columns: minmax(280px, 0.9fr) 1.2fr !important; gap: 16px; align-items: start; }
.cq-final { box-shadow: 0 0 0 2px #99f6e4; }
.cq-source { display: block; margin: 8px 0 0 !important; color: #334155 !important; font-size: 0.84rem; line-height: 1.4; }
.cq-rating, .cq-panel, .cq-metrics, .cq-strengths, .cq-improvements { padding: 16px 18px; }
.cq-rating h2 { margin: 4px 0; font-size: 2rem; letter-spacing: -0.03em; }
.cq-kicker { margin: 0; color: #0f766e !important; font-size: 0.75rem; font-weight: 750; letter-spacing: 0.04em; text-transform: uppercase; }
.cq-score { margin: 0 0 8px; font-size: 1.15rem; font-weight: 700; color: #102033 !important; }
.cq-band-poor { border-top: 4px solid #e11d48; }
.cq-band-average { border-top: 4px solid #d97706; }
.cq-band-good { border-top: 4px solid #059669; }
.cq-band-excellent { border-top: 4px solid #0f766e; }
.cq-band-open { border-top: 4px solid #94a3b8; }
.cq-facts { display: grid; gap: 0; margin: 0; }
.cq-facts div { display: flex !important; justify-content: flex-start !important; gap: 16px !important; align-items: baseline; padding: 8px 0; border-bottom: 1px solid #e2e8f0; }
.cq-facts div:last-child { border-bottom: 0; padding-bottom: 0; }
.cq-facts dt { color: #526277 !important; font-size: 0.84rem; }
.cq-facts dd { margin: 0 !important; font-weight: 700; text-align: left !important; }
.cq-criterion.cq-open { background: #f8fafc !important; }
.cq-panel h2, .cq-rubric h2, .cq-metrics h2, .cq-strengths h2, .cq-improvements h2, .cq-empty h2 { margin: 0 0 10px; font-size: 1rem; }
.cq-panel ul, .cq-strengths ul, .cq-improvements ul { margin: 0; padding-left: 18px; }
.cq-panel li { margin: 8px 0 !important; }
.cq-rubric { margin-top: 20px; }
.cq-criterion-grid, .cq-findings { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.cq-findings { grid-template-columns: 1fr 1fr; gap: 16px; }
.cq-criterion { padding: 16px; margin: 0; }
.cq-criterion p { margin: 8px 0 0 !important; }
.cq-criterion header { display: flex; justify-content: space-between; gap: 8px; align-items: flex-start; }
.cq-criterion h3 { margin: 0; font-size: 0.92rem; }
.cq-criterion span { font-size: 0.75rem; font-weight: 750; white-space: nowrap; }
.cq-high span { color: #047857 !important; }
.cq-mid span { color: #b45309 !important; }
.cq-low span { color: #be123c !important; }
.cq-open span { color: #64748b !important; }
.cq-strengths { background: #f0fdf4 !important; border-color: #bbf7d0; }
.cq-improvements { background: #fffbeb !important; border-color: #fde68a; }
.cq-metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.cq-metric { background: #f8fafc !important; border-radius: 10px; padding: 8px 10px; }
.cq-metric span { display: block; color: #526277 !important; font-size: 0.75rem; font-weight: 650; }
.cq-metric strong { display: block; margin-top: 2px; color: #102033 !important; font-size: 0.98rem; }
.cq-limits { padding: 12px 16px; margin: 12px 0; }
.cq-error { padding: 12px 16px; margin: 10px 0; border: 1px solid #fecdd3; border-radius: 12px; background: #fff1f2 !important; color: #9f1239 !important; }
.cq-switch {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 12px !important;
  padding: 8px 12px !important;
  margin: 8px 0 !important;
}
.cq-switch label {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  min-height: 44px;
  color: #142033 !important;
  font-weight: 700 !important;
  cursor: pointer;
}
.cq-switch input {
  appearance: none !important;
  width: 44px !important;
  height: 24px !important;
  margin: 0 !important;
  border: 0 !important;
  border-radius: 999px !important;
  background: #cbd5e1 !important;
  cursor: pointer;
  position: relative !important;
  flex: 0 0 44px;
}
.cq-switch input::after {
  content: "";
  position: absolute;
  top: 3px;
  left: 3px;
  width: 18px;
  height: 18px;
  border-radius: 999px;
  background: #ffffff;
}
.cq-switch input:checked { background: #0f766e !important; }
.cq-switch input:checked::after { transform: translateX(20px); }
.cq-switch input:focus-visible, .cq-analyze button:focus-visible {
  outline: 2px solid #0f766e !important;
  outline-offset: 2px;
}
.cq-analyze button, button.primary { background: #0f766e !important; color: #ffffff !important; border-radius: 12px !important; font-weight: 750 !important; min-height: 46px; }
.cq-analyze button:hover { background: #0d655e !important; }
@media (prefers-reduced-motion: reduce) {
  .cq-analyze button, .cq-switch input, .cq-switch input::after, .cq-progress span { transition: none !important; }
}
.cq-drop { border: 1.5px dashed #cbd5e1 !important; border-radius: 14px !important; background: #f8fafc !important; min-height: 0 !important; }
.cq-drop button { min-height: 132px; }
@media (max-width: 980px) {
  .cq-result-grid, .cq-compare, .cq-detail, .cq-criterion-grid, .cq-categories, .cq-stages { grid-template-columns: 1fr 1fr; }
  .cq-header { flex-direction: column; align-items: flex-start; }
}
@media (max-width: 680px) {
  .cq-result-grid, .cq-compare, .cq-detail, .cq-criterion-grid, .cq-categories, .cq-stages, .cq-findings, .cq-metric-grid, .cq-files { grid-template-columns: 1fr; }
  .cq-files { flex-direction: column !important; }
  .cq-intro h1 { font-size: 1.35rem; }
  .cq-rating h2 { font-size: 1.6rem; }
  .cq-facts div { flex-wrap: wrap; }
}
"""


def _theme():
    """One light palette for both theme modes, so the page does not change color with the OS theme."""
    theme = gr.themes.Base(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
    )
    light = {
        "body_background_fill": "#eef2f6",
        "body_text_color": "#142033",
        "body_text_color_subdued": "#24344a",
        "background_fill_primary": "#eef2f6",
        "background_fill_secondary": "#ffffff",
        "block_background_fill": "#ffffff",
        "block_border_color": "#e2e8f0",
        "block_label_background_fill": "#ffffff",
        "block_label_text_color": "#142033",
        "block_title_background_fill": "#ffffff",
        "block_title_text_color": "#142033",
        "border_color_primary": "#e2e8f0",
        "input_background_fill": "#f8fafc",
        "input_background_fill_focus": "#ffffff",
        "input_background_fill_hover": "#ffffff",
        "button_primary_background_fill": "#0f766e",
        "button_primary_text_color": "#ffffff",
        "button_secondary_background_fill": "#ffffff",
        "button_secondary_text_color": "#142033",
        "button_secondary_border_color": "#e2e8f0",
    }
    locked = {key: value for key, value in light.items()}
    locked.update({f"{key}_dark": value for key, value in light.items()})
    return theme.set(**locked)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        shown = f"{seconds:.0f}" if float(seconds).is_integer() else f"{seconds:.1f}"
        return f"{shown} s"
    return f"{seconds / 60:.1f} min"


def _format_measure(key: str, value: float, unit: str) -> str:
    if key in {"silence_s", "longest_silence_s"} and value >= 60:
        return f"{value / 60:.1f} min"
    if key.endswith("_ratio"):
        return f"{value * 100:.0f}%"
    if key == "mean_luminance":
        return f"{value:.0f}"
    if unit:
        return f"{value:.1f} {unit}" if isinstance(value, float) and not value.is_integer() else f"{value:g} {unit}"
    return f"{value:g}"


def _error_html(message: str) -> str:
    if not message:
        return ""
    return f'<p class="cq-error">{html.escape(message)}</p>'


def _criterion_badge(status: str, rating: int | None) -> tuple[str, str]:
    if status != "assessed" or rating is None:
        return "cq-open", "Not assessed"
    if rating >= 2:
        return "cq-high", "2 / 2"
    if rating == 1:
        return "cq-mid", "1 / 2"
    return "cq-low", "0 / 2"


def _findings_html(strengths: list[str], improvements: list[str]) -> str:
    if strengths:
        strength_body = "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in strengths) + "</ul>"
    else:
        strength_body = "<p>This report does not contain enough evidence for strengths or improvements.</p>"
    if improvements:
        improve_body = "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in improvements) + "</ul>"
    else:
        improve_body = "<p>No evidence-backed improvement was recorded. Criteria that could not be assessed are not listed as failures.</p>"
    return (
        '<section class="cq-findings">'
        f'<article class="cq-strengths"><h2>Strengths</h2>{strength_body}</article>'
        f'<article class="cq-improvements"><h2>Areas for improvement</h2>{improve_body}</article>'
        "</section>"
    )


def _metrics_html(measurements: dict[str, object]) -> str:
    cards = []
    duration = measurements.get("duration_s")
    if isinstance(duration, (int, float)):
        cards.append(("Duration", _format_duration(float(duration))))
    share = measurements.get("student_turn_share")
    if isinstance(share, (int, float)):
        cards.append(("Student participation", f"{share * 100:.0f}% of turns"))
    homework = measurements.get("homework_mentions")
    if isinstance(homework, int):
        cards.append(("Homework mentions", str(homework)))
    loudness = measurements.get("loudness_lufs")
    if isinstance(loudness, (int, float)):
        cards.append(("Integrated loudness", f"{loudness:.1f} LUFS"))
    for key, label, unit in (
        ("silence_s", "Silence", "s"),
        ("longest_silence_s", "Longest silence", "s"),
        ("mean_luminance", "Mean luminance", ""),
        ("dark_frame_ratio", "Dark-frame ratio", ""),
        ("static_frame_ratio", "Static-frame ratio", ""),
        ("frame_count", "Sampled frames", ""),
        ("teacher_words_per_minute", "Teacher pace", "words/min"),
    ):
        value = measurements.get(key)
        if isinstance(value, (int, float)):
            cards.append((label, _format_measure(key, value, unit)))
    body = "".join(
        f'<div class="cq-metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    return (
        '<section class="cq-metrics"><h2>Measured signals</h2>'
        '<p>These describe the recording. They are not the teaching score.</p>'
        f'<div class="cq-metric-grid">{body}</div></section>'
    )


def _scrub_report(report: Report, names: list[str]) -> Report:
    criteria = [
        replace(item, explanation=_scrub(item.explanation, names))
        for item in report.criteria
    ]
    return replace(
        report,
        reliability_notes=[_scrub(note, names) for note in report.reliability_notes],
        criteria=criteria,
        strengths=[_scrub(item, names) for item in report.strengths],
        improvements=[_scrub(item, names) for item in report.improvements],
    )


def _hybrid_report(video, transcript, names, measure, client, stills_fn, workspace, local=None, sample_frames=None):
    sampled = []
    if sample_frames is not None:
        try:
            sampled = list(sample_frames(str(video), 0.0, workspace) or [])
        except Exception:
            sampled = []
    measurement_failed = False
    try:
        media = measure(str(video))
    except MediaError:
        media = None
        measurement_failed = True
    if sample_frames is not None and media is not None and not sampled:
        try:
            sampled = list(sample_frames(str(video), media.duration_s, workspace) or [])
        except Exception:
            sampled = []
    if local is not None:
        cues = list(local.cues)
    else:
        cues = [] if transcript is None else parse_vtt(transcript.read_text(encoding="utf-8"))
    pseudonymised = pseudonymise(cues)
    features = extract_features(
        pseudonymised.cues,
        duration_s=None if media is None else media.duration_s,
    )
    if local is not None and not local.reliable:
        features = replace(
            features,
            transcript_reliable=False,
            reliability_reasons=[local.reason, *features.reliability_reasons] if local.reason else list(features.reliability_reasons),
        )
    frames = _stills(video, media, workspace, stills_fn, sampled)
    pack = EvidencePack(
        cues=cues,
        features=features,
        media=media,
        frames=frames,
        forbidden_names=names,
    )
    rules_report = score_rules(features, media)
    judge_client = client if client is not None else OllamaCloudClient(permitted=True)
    ai_report = RubricJudge(judge_client).score(pack, rules_report)
    extra: list[str] = []
    if measurement_failed:
        extra.append("FFmpeg could not measure the recording. Technical quality was not assessed.")
    if local is not None and local.cues and not local.speaker_labelled:
        extra.append("Local transcription has no speaker labels. Student participation was not measured.")
    supplied = transcript is not None and local is None

    def _annotate(report: Report) -> Report:
        if extra:
            report = replace(report, reliability_notes=[*extra, *report.reliability_notes])
        return attach_transcript_metadata(report, local, supplied=supplied)

    return _annotate(rules_report), _annotate(ai_report)


def _stills(video: Path, media: MediaStats | None, workspace: Path, stills_fn, sampled=None) -> list[FrameStill]:
    if stills_fn is not None:
        return list(stills_fn(str(video), media, workspace))
    # Timestamps stay with the local evidence. Unblurred image bytes are not attached.
    return [FrameStill(timestamp_s=frame.timestamp_s, image_base64=None) for frame in sampled or []]


def extract_stills(video_path: str, media: MediaStats | None, workspace: Path) -> list[FrameStill]:
    """Sample PNG stills locally. The upload path does not call this until frames are face-blurred."""
    if media is None or media.duration_s <= 0 or shutil.which("ffmpeg") is None:
        return []
    frames: list[FrameStill] = []
    for fraction in (0.1, 0.5, 0.9):
        timestamp = media.duration_s * fraction
        target = workspace / f"frame-{len(frames)}.png"
        try:
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    str(target),
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode != 0 or not target.is_file():
            continue
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
        frames.append(FrameStill(timestamp_s=timestamp, image_base64=encoded))
    return frames


def _speaker_names(transcript: Path | None) -> list[str]:
    if transcript is None or not transcript.is_file():
        return []
    names: list[str] = []
    for cue in parse_vtt(transcript.read_text(encoding="utf-8")):
        if not cue.speaker:
            continue
        names.append(cue.speaker)
        names.extend(token for token in cue.speaker.split() if token)
    return names


def _scrub(text: str, names: list[str]) -> str:
    return scrub_names(text, names)


def _oversized(path: Path, limit: int, kind: str) -> str | None:
    """Reject from stat() so the file is not read or copied into the workspace."""
    size = path.stat().st_size
    if size <= limit:
        return None
    return (
        f"The {kind} is {_format_size(size)}, above the {_format_size(limit)} upload limit, "
        "so it was not copied. This local tool allows up to 8 GiB so a 40–90 minute class "
        "recording (about 12 Mbps for 90 minutes) can be analysed."
    )


def _format_size(num_bytes: int) -> str:
    gib = 1024**3
    if num_bytes >= gib and num_bytes % gib == 0:
        return f"{num_bytes // gib} GiB"
    return f"{num_bytes} bytes"


def _as_path(value) -> Path:
    name = getattr(value, "name", None)
    return Path(name if isinstance(name, str) else str(value))


def _mark(progress, on_step, stage: int, fraction: float, message: str) -> None:
    _progress(progress, fraction, message)
    if on_step is not None:
        on_step(stage, fraction, message)


def _progress(progress, fraction: float, message: str) -> None:
    if progress is None:
        return
    try:
        progress(fraction, desc=message)
    except TypeError:
        try:
            progress(fraction, message)
        except Exception:
            return
    except Exception:
        return


if __name__ == "__main__":
    main()
