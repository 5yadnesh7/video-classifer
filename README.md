# ClassQuality AI

Assesses one online class recording as **Poor**, **Average**, **Good**, or **Excellent**, and explains the result from the evidence it could score.

## How to run

Requirements: Python 3.12, and both `ffmpeg` and `ffprobe` on `PATH`.

From this directory on Windows:

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python.exe -m class_quality.app
```

Open http://127.0.0.1:7860. Upload an MP4. A `.vtt` transcript is optional. If it is omitted, speech is transcribed on this computer. The page binds to localhost and does not create a public share link.

The AI judge runs only when `.env` contains `OLLAMA_CLOUD_PERMITTED=1` and an `OLLAMA_API_KEY`. Any other permission value keeps the run on this computer. `.env` is gitignored. Do not commit it, and do not commit class recordings.

Rules-only command line:

```text
.venv\Scripts\python.exe -m class_quality --video path\to\recording.mp4 --vtt path\to\transcript.vtt
```

`--vtt` is optional. The command does not call the network.

## Approach

The scorer has two layers, and the page shows both.

The local rules read the transcript and a few measurements from the file: whether the class opens with a greeting, how much of the talk is from students, whether it closes, whether homework is mentioned, and whether the audio is present and loud enough to use. Fixed weights turn those ratings into a score. Excellent starts at 80, Good at 65, Average at 50, and Poor is below 50. A criterion with no evidence is left out of the score. It is not marked zero.

Tone, whether the teacher is on camera, and the branded background are not scored from text. On the sample classes, those were often why a class was called weak. A transcript-only score therefore runs high on some weaker classes. Checked once on the ten labelled recordings, the rules were exact on 5 and within one class on 9. The weights were not fitted to that sample.

With the AI switch on, and only when cloud use is permitted, Ollama Cloud (`gemma4:31b`) scores the same criteria from the pseudonymised transcript and the measured numbers. The recording, the audio, and the stills stay on this computer. The judge may not raise a rating above the local measurement, and a criterion it leaves blank is not treated as full marks.

What worked was an explainable local score that runs with no network call. What failed was treating a partial checklist as a class: early judge replies were loose JSON, or a few perfect marks that looked like 100. The limits are English-only transcripts, no separate check of camera or background until faces can be blurred, and a small labelled set that is not large enough to retune the weights.

## Privacy

These are children's classes. Do not publish the recordings, put them in this repository, or expose them through a public app link. Process them only with services the organiser has permitted.
