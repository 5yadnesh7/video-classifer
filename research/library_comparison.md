# Library comparison (traditional / rule-based tooling)

Access date for all links: **2026-10-22**. "Verified" means checked against the official README, docs or PyPI page on that date. "Measured" means run locally on this dataset. Anything else is marked **unverified**.

Local environment found: Python 3.14.0 and 3.12 installed, FFmpeg 8.1.1 (full build), OpenCV 4.13.0, NumPy 2.4.4, Pydantic 2.13, Ollama CLI client 0.40.8 (local server 0.21.2), CPU-only (Intel i7-1370P, 14 cores, 32 GB RAM, Intel Iris Xe GPU). Nothing was installed for this research.

## 1. Media decoding and frame sampling

| Library | What it does well | Limitations | Verdict |
|---|---|---|---|
| **FFmpeg** (CLI, 8.1.1) | Decodes anything. Built-in filters give audio stats without Python deps: `silencedetect`, `ebur128` (loudness/LRA), `volumedetect`, `astats`. Fast frame extraction (`-ss` seek + `fps=` or `select=`). Extracts 16 kHz mono WAV for ASR. **Measured:** silence + loudness on a 60-min file in ~10 s. | Text-output parsing is brittle. Must be on the host. | **Use**: the backbone for audio extraction and frame sampling. |
| **OpenCV** (`opencv-python-headless`, 5.0.0.93 on PyPI) | Frame reading, resize, colour/luminance stats, frame differencing. Includes the YuNet face detector (`cv2.FaceDetectorYN`, needs a small ONNX model file). **Measured:** 1 frame/30 s sampling + stats on all 12 files ≈ 4.5 min total. | Seeking in long H.264 is slower than FFmpeg. **The bundled Haar face cascade performed poorly here** (2–41% face-frame rate). | **Use** for frame statistics. Use YuNet rather than Haar for faces (YuNet accuracy on this layout **unverified**). |
| **PySceneDetect** 0.7.2 | Shot/scene-change detection (content, adaptive, threshold detectors). | Zoom recordings have few hard cuts; layout switches (gallery ↔ screen share) are its main use here. | Optional. Simple frame-difference stats already capture most of this. |
| **PyAV** | Pythonic FFmpeg bindings (bundled by faster-whisper). | Lower-level API. | Comes free with faster-whisper; no need to use it directly. |

Links: [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html) · [OpenCV PyPI](https://pypi.org/project/opencv-python-headless/) · [PySceneDetect PyPI](https://pypi.org/project/scenedetect/)

## 2. Audio quality, silence, speech activity, volume, pace

| Library | Capabilities (verified) | Limitations | Verdict |
|---|---|---|---|
| **Silero VAD** 6.2.3 ([README](https://github.com/snakers4/silero-vad)) | Neural voice-activity detection. ~2 MB model, <1 ms per 30 ms chunk on one CPU thread, 8/16 kHz, trained on 6000+ languages, MIT licence, no telemetry. `get_speech_timestamps()` returns speech segments. | Requires `torch` (or ONNX runtime with your own I/O). Detects speech only, not who is speaking. | **Use**: speech ratio, dead-air detection, and pre-filtering for ASR. Already bundled inside faster-whisper's `vad_filter`. |
| **librosa** 1.0.0 | RMS energy, pitch (`pyin`), spectral features, onset/tempo. | Slow on 60-min audio at full rate; downsample and window. Pitch on mixed multi-speaker audio is noisy. | Optional: pitch variability and energy as a weak proxy for "energetic delivery". |
| **FFmpeg `ebur128` / `silencedetect`** | Integrated loudness, loudness range, silence intervals. | Threshold-based; can't tell music or noise from speech. | **Use** for cheap audio-quality checks. **Measured:** loudness is *not* discriminative between labels here. |
| **webrtcvad** | Very light VAD. | Noticeably less accurate than Silero in noisy audio (commonly reported; **unverified** here). | Skip; Silero is already a dependency. |
| **pyannote.audio** ([README](https://github.com/pyannote/pyannote-audio)) | State-of-the-art speaker diarisation (`community-1` pipeline). | Gated Hugging Face model (accept terms plus token). Heavy torch stack. Speed benchmark is on an H100 GPU (31 s per hour), so CPU speed is **unverified** and likely many minutes. Optional telemetry (can be disabled). | **Defer.** Only needed if the test recording has no VTT and teacher/student talk ratio is essential. A cheaper fallback is a single-speaker "teacher voice" heuristic. |
| **Speaking pace** | Words per minute = ASR word timestamps ÷ speech time. | Depends on ASR quality; meaningless for mis-transcribed languages (see the Malayalam demo). | Derive from transcript; no extra library. |

## 3. Speech-to-text

| Library | Capabilities (verified) | Limitations | Verdict |
|---|---|---|---|
| **faster-whisper** 1.2.1 ([README](https://github.com/SYSTRAN/faster-whisper), [PyPI](https://pypi.org/project/faster-whisper/)) | CTranslate2 reimplementation of Whisper, "up to 4× faster than openai/whisper" at equal accuracy. INT8 on CPU. Built-in Silero VAD filter. Word timestamps. Language detection. Batched pipeline. MIT. Decodes via bundled PyAV (no system FFmpeg needed). Official CPU benchmark: **13 min audio, `small` model, int8, 8 threads on an i7-12700K → 1 m 42 s (51 s batched)**. | **No speaker diarisation.** Extrapolated: ~6–7 min for 50 min of audio with `small` int8 on 8 threads, and much slower on a 2-vCPU free host (**extrapolation, unverified**). Malayalam is in Whisper's language list, but accuracy on Indian-English / Malayalam code-mixed child speech is **unverified** and likely weak. | **Use** when no VTT is supplied. Start with `small` or `base` int8 with VAD. Consider transcribing only sampled windows (first 10 min, last 8 min, plus a few mid-class windows) to bound latency. |
| **openai-whisper** | Reference implementation. | ~4× slower; needs system FFmpeg. | Skip. |
| **whisper.cpp** | Fastest CPU option in the faster-whisper benchmark table (2 m 05 s fp32; 1 m 45 s with OpenVINO). | C++ build/binary management; extra integration work. | Fallback only if Python wheels fail. |
| **WhisperX** | faster-whisper plus alignment plus pyannote diarisation. | Inherits pyannote's gated model and heavy stack. | Defer (same reasoning as pyannote). |
| **Provided Zoom VTT** | Already speaker-labelled; zero compute. | Not guaranteed at test time. English-only transcriber garbles other languages. Contains real names. | **Use when present**, after pseudonymising names. |

Python 3.14 wheels exist on PyPI for `ctranslate2` 4.8.3, `onnxruntime` 1.31.0, `torch` 2.14.1 and `scikit-learn` 1.9.1 (checked via the PyPI JSON API). Installing them was **not** tested. **Recommendation: build on Python 3.12** (already installed) to match common hosting images and avoid bleeding-edge wheel gaps.

## 4. Face, camera and visual signals

| Library | Capabilities | Limitations | Verdict |
|---|---|---|---|
| **OpenCV Haar cascade** | Bundled, zero download. | **Measured: unreliable here.** | Don't rely on it. |
| **OpenCV YuNet** (`FaceDetectorYN`) | Small CNN face detector, runs in OpenCV with an ONNX file. | Needs a model file download. Accuracy on small Zoom tiles **unverified**. | **Preferred local detector** for "teacher camera on / face visible". |
| **MediaPipe** 1.1.1 ([PyPI](https://pypi.org/project/mediapipe/)) | Face detection/landmarks, on-device inference. | Its privacy notice says Tasks APIs **send performance/usage metrics to Google** (not the media itself). Pulls in `opencv-contrib-python` and `matplotlib`. | Acceptable, but YuNet avoids the telemetry question. |
| **Luminance / colour stats** (NumPy) | Lighting adequacy, black frames (camera off), static frames. | Global stats mix all tiles; need layout-aware crops for "teacher tile". | **Use** as cheap evidence. |
| **Branded background detection** | Could use template matching against a reference "8x8" background crop, or a VLM question. | Needs a reference image I have not extracted (frames not viewed). Virtual backgrounds vary. | Prefer a VLM yes/no on a few face-blurred teacher-tile crops, **if a cloud vision model is permitted**. Otherwise template match. |

## 5. Lesson structure and interaction signals

No single library does this. It is assembled from the transcript plus simple rules:

| Signal (maps to feedback theme) | How to compute | Evidence quality on this dataset |
|---|---|---|
| Intro / rapport | Greeting, name and "how are you" patterns plus student turns in the first 5–8 min, before lesson keywords | Plausible; small-sample counts are noisy |
| New joiners introduced | Student self-introduction turns early; LLM check | Unverified |
| Student participation | Student share of turns or talk time (needs speaker labels) | **Strongest numeric signal measured** (Excellent ≈ 0.55 → Average ≈ 0.30), but Poor recordings don't fit, so it is not sufficient alone |
| Teacher questions | `?` count and question words per 10 min | Weak alone |
| Homework | Keyword regex (`homework`, `HW`, `assignment`, `practice`) | **Matched the feedback in all checkable cases** |
| Conclusion | Recap/goodbye patterns in the last 5 min | Noisy |
| Dead air | Silero VAD gaps longer than N s, discounted when context suggests independent work | Ambiguous (Good recording had long silences) |
| Energy / tone | Exclamations and praise words; pitch variability; LLM judgment | Weak numerically; better judged by an LLM with transcript evidence |

**Transcript parsing:** `webvtt-py` 0.5.1 ([PyPI](https://pypi.org/project/webvtt-py/)) or the ~30-line regex parser already used during inspection.

## 6. Transparent scoring and traditional ML

| Library | Use | Limitations |
|---|---|---|
| **Plain Python + Pydantic** | Rubric config (YAML/dict) → weighted sub-scores → class thresholds. Every point is traceable to a measured value and timestamp. | Weights are hand-set and must be justified from the feedback themes, not fitted to 10 files. |
| **scikit-learn** 1.9.1 | Logistic regression, ordinal-style models, `LeaveOneOut` CV, permutation importance. | **10 labelled samples (2/2/2/4) is far too few** to train or validate a classifier. Any accuracy number would be noise and leakage-prone (teacher identity, batch). Usable only as a sanity check that rule features order the labels, not as the decision maker. |

## 7. App / UI

| Option | Notes |
|---|---|
| **Gradio** 6.30.0 | File upload plus JSON/Markdown output in a few lines. Native Hugging Face Spaces hosting. Upload size limits and the free CPU tier specs need checking (**unverified**). |
| **Streamlit** 1.66.1 | Equally simple. Default upload limit is configurable (`server.maxUploadSize`). Streamlit Community Cloud resources are modest (**unverified**). |

Either is acceptable; pick the one whose host gives the most CPU/RAM for transcription. Hosting must **not** persist uploads and must **not** ship any sample recording as a demo example.

## Summary recommendation

Minimal, sufficient stack: **FFmpeg + OpenCV (+ YuNet) + Silero VAD (via faster-whisper) + faster-whisper (only when no VTT) + plain-Python rubric scoring + Gradio**. Defer pyannote, librosa pitch, and any trained classifier.
