# Dataset inventory

Inspection date: 2026-10-22. All measurements below were computed **locally** (FFmpeg 8.1.1 `ffprobe`/filters, OpenCV 4.13 on sampled frames, and a plain-Python WebVTT parser). No recording, frame, audio clip or transcript was uploaded anywhere.

> **Confidentiality.** This document intentionally contains no teacher names, child names, transcript quotes or images. Recordings are referred to by folder label and batch ID only. The organiser's feedback is paraphrased as rubric themes. Review before publishing even so: batch IDs and dates are internal identifiers.

## 1. What is in the folder

```
dataset/
  feedbacks.pdf                 organiser's written feedback per recording (labels + reasons)
  excellent/       2 x (.mp4 + .vtt)
  good/            2 x (.mp4 + .vtt)
  average_/        2 x (.mp4 + .vtt)
  weak/            2 x (.mp4 + .vtt)
  quality issue_/  2 x (.mp4 + .vtt)
  demo_/           2 x (.mp4 + .vtt)   (no quality label)
```

12 recordings (≈ 1.72 GB total, ≈ 10.5 hours), 12 matching transcripts, 1 feedback PDF.

## 2. Labels: what exists and what doesn't

Labels **do exist**, in two forms:

1. **Folder name** (5 quality folders + 1 demo folder).
2. **`feedbacks.pdf`**, which lists each of the 10 non-demo recordings under a heading (*Excellent / Good / Average / Weak / Known quality issues*) with a one-line reviewer comment. Batch IDs and dates match the folder contents one-to-one.

Mapping to the required 4-class output (Poor / Average / Good / Excellent). This is my interpretation and still needs confirming with the organiser:

| Folder | Feedback heading | Proposed target class | n |
|---|---|---|---|
| `excellent` | Excellent | Excellent | 2 |
| `good` | Good | Good | 2 |
| `average_` | Average | Average | 2 |
| `weak` | Weak | Poor | 2 |
| `quality issue_` | Known quality issues | Poor (feedback says the tutor was "informed about the quality issues several times") | 2 |
| `demo_` | — (not in feedback PDF) | **Unlabelled**: use only for qualitative smoke tests | 2 |

**Labelled sample size: 10 recordings (2/2/2/4).** Each label comes from one reviewer comment. There is no inter-rater information, no numeric score, and no per-timestamp annotation.

### Rubric themes recovered from the feedback (paraphrased)

These are the most valuable part of the dataset. They tell us what the organiser's reviewers actually look for:

| Theme | Appears as positive in | Appears as criticism in |
|---|---|---|
| Intro / rapport building at the start | Excellent, Good | Weak, Quality-issue ("went directly into content") |
| Giving new joiners time to introduce themselves | — | Weak (both) |
| Including everyone; calling on students individually | Excellent, Good, Average | — (present even in some Poor) |
| Homework assigned | Excellent, Good, Average | Good/Average ("should give HWs") |
| Branded "8x8" virtual background kept | most | Quality-issue ("no background") |
| Teacher visible: camera on, face visible, adequate lighting | — | Average ("sit where there is light"), Weak ("camera turned off"), Quality-issue ("can't see his face") |
| Energetic, pleasant, happy tone | Excellent | Average, Weak, Quality-issue ("need more energetic/pleasant/happier") |
| Proper conclusion | — | Weak |
| AV clarity | — | Average ("clarity issues while keeping the background") |
| Topic-related videos / puzzles / activities | Excellent (suggestion), Average (puzzles) | — |

Implication: quality is judged mainly on **teaching behaviour and on-camera professionalism**, not on chess content correctness. Several criteria are **visual** (background, camera on, face visible, lighting), so a transcript-only system will miss them.

## 3. Media properties (ffprobe)

All files: MP4 container, H.264 video at 25 fps, AAC stereo audio at 48 kHz. These are Zoom-style cloud recordings.

| Label | Batch | Size (MB) | Duration (min) | Resolution | Total bitrate (kbps) |
|---|---|---|---|---|---|
| average | B435 | 51.3 | 37.9 | 1920×1080 | 189 |
| average | B516 | 324.6 | 43.6 | 1920×1080 | 1041 |
| demo | Malayalam | 127.5 | 51.7 | 1920×1080 | 345 |
| demo | English | 160.3 | 44.1 | 1920×1080 | 509 |
| excellent | B472 | 217.1 | 56.0 | 1920×1080 | 542 |
| excellent | B491 | 175.4 | 53.9 | 1920×1030 | 455 |
| good | B435 | 73.9 | 44.5 | 1920×1080 | 232 |
| good | B552 | 124.7 | 45.8 | 1920×1080 | 380 |
| quality issue | B392 | 157.2 | 60.5 | 1920×1080 | 364 |
| quality issue | B441 | 110.5 | 89.4 | 1920×1020 | 173 |
| weak | B482 | 157.7 | 59.2 | 1920×1080 | 372 |
| weak | B558 | 73.4 | 46.2 | 1920×1080 | 222 |

Range: 38–89 min, 51–325 MB. **An unseen test file is likely ~45–60 min and 50–350 MB.** Upload size and processing time are real design constraints.

## 4. Transcripts (.vtt)

- WebVTT with numbered cues; every cue is `Speaker Name: utterance`. **All cues are speaker-labelled** (0 unlabelled), so transcripts double as **speaker diarisation**.
- Speaker labels are **real names of teachers and children**. They must be pseudonymised (e.g. `TEACHER`, `STUDENT_1`) before any text leaves the machine.
- Two caveats found:
  - **"Speaker with most talk time" is not reliably the teacher.** In one Good recording, a student had more words than the teacher; in the English demo, the teacher is not the speaker with most turns. Identifying the teacher needs a better rule, e.g. the speaker asking the most questions, or the speaker whose name matches the meeting host if that is available.
  - **The Malayalam demo transcript is Malayalam speech forced into English words** (0% non-Latin characters; teacher speech rate ≈ 72 words/min vs 130–186 elsewhere). The built-in transcriber assumed English. Transcript-based signals are unreliable for non-English or code-mixed classes.
- **It is unknown whether the unseen test recording will come with a `.vtt`.** The brief only says "a recording". The design must work from the MP4 alone and treat a VTT as an optional accuracy boost.

### Aggregate transcript statistics (no content)

| Label | Batch | Speakers | Words | Student share of turns | Teacher wpm | Homework mentions (teacher) | Gaps > 60 s |
|---|---|---|---|---|---|---|---|
| excellent | B472 | 7 | 7233 | 0.56 | 186 | 2 | 0 |
| excellent | B491 | 9 | 5606 | 0.55 | 158 | 0 | 0 |
| good | B435 | 8 | 2695 | 0.46 | 183 | 0 | 4 |
| good | B552 | 5 | 4523 | 0.49 | 177 | 3 | 1 |
| average | B435 | 5 | 3343 | 0.29 | 139 | 0 | 2 |
| average | B516 | 6 | 2872 | 0.30 | 132 | 7 | 1 |
| weak | B482 | 6 | 2039 | 0.40 | 166 | 0 | 9 |
| weak | B558 | 4 | 4273 | 0.43 | 164 | 0 | 1 |
| quality issue | B392 | 6 | 4785 | 0.34 | 154 | 0 | 3 |
| quality issue | B441 | 11 | 5848 | 0.43 | 151 | 0 | 10 |
| demo | English | 3 | 5787 | 0.52 | — | 0 | 0 |
| demo | Malayalam | 2 | 2391 | 0.35 | — | 0 | 1 |

"Teacher" here = the speaker who asks the most questions. The table is computed from the cues, and wpm uses the teacher's summed cue durations.

**Early, small-sample observations (n = 10; not statistically meaningful):**
- **Homework mentions agree with the feedback in every case the feedback comments on.** Recordings described as "given HWs" have 2–7 mentions; those told "should give HWs" have 0. This is a cheap, explainable rule.
- **Student share of turns** separates Excellent (≈ 0.55) > Good (≈ 0.47) > Average (≈ 0.30). The four Poor recordings (0.34–0.43) sit *above* Average. Their feedback mostly cites camera/background/tone problems, which the transcript can't see. **This is a concrete argument for combining transcript and visual evidence.**
- **Addressing students by name is *not* discriminative.** Both quality-issue recordings address every student by name many times (the feedback agrees: "interaction was there").

## 5. Local audio/video measurements

Audio: FFmpeg `silencedetect` (−40 dB, ≥ 5 s) and `ebur128`. Video: one frame every 30 s, mean luminance, OpenCV Haar face detector, and frame-to-frame difference.

| Label | Batch | Silence ≥5 s total (min) | Longest silence (s) | Loudness (LUFS) | Loudness range (LU) | Mean luminance | Haar face-frame rate |
|---|---|---|---|---|---|---|---|
| excellent | B472 | 1.0 | 21 | −16.5 | 12.5 | 104.5 | 0.15 |
| excellent | B491 | 4.3 | 49 | −17.9 | 10.7 | 83.0 | 0.21 |
| good | B435 | 22.1 | 387 | −15.7 | 18.6 | 62.9 | 0.10 |
| good | B552 | 3.4 | 61 | −17.6 | 11.6 | 86.9 | 0.07 |
| average | B435 | 8.3 | 109 | −17.6 | 17.5 | 79.9 | 0.03 |
| average | B516 | 11.2 | 51 | −15.9 | 21.2 | 50.9 | 0.41 |
| weak | B482 | 24.2 | 133 | −20.2 | 23.1 | 65.0 | 0.34 |
| weak | B558 | 5.8 | 139 | −19.3 | 13.0 | 73.2 | 0.08 |
| quality issue | B392 | 11.3 | 115 | −17.6 | 10.6 | 52.9 | 0.23 |
| quality issue | B441 | 33.3 | 100 | −16.1 | 15.6 | 86.0 | 0.02 |
| demo | Malayalam | 10.1 | 439 | −19.1 | 8.5 | 70.9 | 0.16 |
| demo | English | 1.5 | 44 | −18.2 | 9.9 | 72.5 | 0.32 |

**Observations:**
- **Loudness is not discriminative** (all −15.7 to −20.2 LUFS). It is useful only as a sanity check for broken audio.
- **Silence is ambiguous.** Both Excellent recordings have little dead air, but a *Good* recording has the second-longest single silence (likely silent puzzle-solving time). Silence must be interpreted in context, not penalised blindly.
- **The Haar face detector is unreliable on this layout** (2–41% of frames). It is probably failing on small gallery tiles or screen-share frames rather than reflecting real face absence. "Teacher visible" needs a stronger detector (OpenCV YuNet / MediaPipe) or a vision-language model, plus an understanding of the Zoom layout.
- Processing cost: these statistics took ≈ 15–37 s per recording on a laptop CPU (i7-1370P, no GPU).

## 6. What is missing

- No numeric scores, no confidence, no second rater; a single short comment per recording.
- No labels for the two demo recordings.
- No per-segment annotations (e.g. when the intro ends or when homework is given).
- No information on whether test recordings will include a transcript, or what language they will be in.
- No statement of **which external services the organiser permits** for processing recordings. The brief says to use only "services specifically permitted by the organiser". **This is a blocker for any cloud model call until confirmed.**
- Possible confounds: the same batch ID (B435) appears under both Good and Average with different teachers. One demo teacher also appears in an Excellent recording. Teacher identity therefore leaks into the labels, and any classifier must never see names.
- I did not visually inspect any frame, so the Zoom layout (gallery vs speaker view vs screen share) is **unverified**.
