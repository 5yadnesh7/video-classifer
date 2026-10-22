# Approach comparison and evaluation design

Context that drives every trade-off (see `dataset_inventory.md`):

- **10 labelled recordings** (2 Excellent, 2 Good, 2 Average, 4 Poor), one short reviewer comment each, plus 2 unlabelled demos.
- The reviewer comments give an **explicit, recoverable rubric**: intro/rapport, welcoming new joiners, including everyone, homework, branded background, teacher camera/face/lighting, energetic and pleasant tone, conclusion, AV clarity.
- Several rubric items are **visual**; several are **conversational**; almost none are about audio loudness.
- Test inputs are 40–90 min MP4s, 50–350 MB. A transcript **may or may not** accompany them.
- No Ollama Cloud model takes video or audio. Cloud use requires organiser permission, which is not yet given.

## A. Rule-based scoring on extracted features

Features (silence, VAD speech ratio, student turn share, homework keywords, greeting/conclusion patterns, luminance, face presence, camera-off frames) are mapped to rubric points with hand-written thresholds.

- **Strengths:** fully transparent, deterministic and offline (no privacy issue). Cheap. Every point traces to a number and timestamp. Already partly validated: homework keywords matched the feedback in every checkable case.
- **Weaknesses:** can't judge *tone*, *quality of rapport* or *whether questions are meaningful*. Keyword rules are brittle to phrasing and ASR errors. Without speaker labels (no VTT), participation metrics need diarisation. Thresholds tuned on 10 files will overfit.
- **Implementation time:** ~2–3 h for a credible feature set.
- **Compute:** CPU only. FFmpeg/OpenCV stats ≈ 15–40 s per recording (measured). ASR adds minutes when no VTT is supplied.
- **Likely failure modes:** silent puzzle time penalised as dead air (a *Good* recording had 22 min of ≥5 s silences). A teacher who addresses everyone by name but is cold or off-camera scores high (both quality-issue recordings did this). Non-English classes break text rules. Haar face detection fails on Zoom tiles (measured).

## B. AI model evaluates transcript + visual evidence (free-form)

Send a pseudonymised transcript and a few frames to a VLM and ask "rate this class and explain".

- **Strengths:** captures tone, rapport and pedagogy. Natural, specific explanations. Little code.
- **Weaknesses:** no fixed criteria, so ratings drift and are hard to defend. Models tend to grade generously and produce generic praise ("engaging session"). Can hallucinate events. Output parsing is fragile (no schema enforcement on Ollama Cloud). Requires cloud permission.
- **Implementation time:** ~1–1.5 h.
- **Compute:** local extraction (frames + transcript) plus one cloud call, ≈ $0.003–0.08 per class.
- **Likely failure modes:** central-tendency bias (everything is "Good"). Sensitivity to prompt wording. Missing visual problems if the sampled frames happen to look fine. Inconsistent results across runs.

## C. Hybrid: measured features + AI reasoning

Local code measures what is measurable (camera on/off, lighting, silence timeline, participation ratios, homework mentions, segment timestamps). The model receives those **numbers plus transcript excerpts** and reasons about what numbers can't capture (rapport, tone, inclusion quality), citing timestamps.

- **Strengths:** each component does what it is good at. Explanations are grounded in measured facts, so hallucination is easier to detect. Degrades gracefully: if the cloud is unavailable, a features-only report from A remains.
- **Weaknesses:** more moving parts. The model may over-weight or ignore numbers unless told how to use them.
- **Implementation time:** ~3–4 h (A's extractor plus a prompt and parser).
- **Compute:** same as A plus one cloud call.
- **Likely failure modes:** the model contradicting the measured features (mitigated by validation rules). Errors from upstream ASR or face detection propagating into the reasoning.

## D. AI-as-judge against an explicit rubric

The model scores each rubric criterion separately (e.g. 0/1/2) with **required evidence** (timestamp plus short paraphrase), returned as JSON. The final class is computed **deterministically** from the criterion scores.

- **Strengths:** the rubric comes from the organiser's own feedback, so it is aligned with how the labels were made. Per-criterion output is easy to validate against the feedback text, which gives many more evaluation points than 10 overall labels. Stable aggregation. Specific, non-generic explanations by construction.
- **Weaknesses:** still depends on what evidence the model is shown. Criteria that need vision must be answered from frames or measured features. Rubric weights are a judgment call.
- **Implementation time:** ~1 h on top of C (rubric prompt, JSON validation, aggregation).
- **Compute:** one call (chat) or a few calls (Clef Decision API for probabilistic per-criterion scores).
- **Likely failure modes:** the model filling every criterion with "partial" (hedging), and evidence citations that don't exist. Mitigate by validating cited timestamps against the transcript and rejecting or retrying.

## E. Traditional ML classifier on labelled examples

Train logistic regression / gradient boosting on per-recording feature vectors.

- **Strengths:** objective weights; standard methodology.
- **Weaknesses:** **not feasible as a decision maker with n = 10** (2/2/2/4 per class). Leave-one-out accuracy would swing 10 points per file, and a feature-rich model will memorise. Teacher and batch identity leak across classes (the same batch appears in two classes; one teacher appears in a demo and an Excellent recording). It also gives **no natural-language explanation**.
- **Implementation time:** ~1 h, but most of it would produce misleading numbers.
- **Compute:** trivial.
- **Likely failure modes:** overfitting, unstable feature importance, and confident wrong answers on unseen teachers.
- **Legitimate limited use:** a sanity check that the rule features *order* the labels (rank correlation), never the final classifier.

## Scorecard

| | A Rules | B Free-form AI | C Hybrid | D Rubric judge | E ML |
|---|---|---|---|---|---|
| Processes unseen recordings | ✅ | ✅ | ✅ | ✅ | ✅ (badly) |
| Specific, evidence-backed explanation | ◐ (mechanical) | ◐ (often generic) | ✅ | ✅ | ❌ |
| Covers visual rubric items | ◐ (stats only) | ◐ (frames only) | ✅ | ✅ (with C's features) | ◐ |
| Covers tone/rapport | ❌ | ✅ | ✅ | ✅ | ❌ |
| Stability / reproducibility | ✅ | ❌ | ◐ | ✅ (deterministic aggregation) | ✅ |
| Works without cloud permission | ✅ | ❌ | ◐ (falls back to A) | ❌ (unless local LLM) | ✅ |
| Build time within 4–5 h | ✅ | ✅ | ✅ | ✅ (on top of C) | ✅ |
| Risk of overfitting the 10 samples | Medium | Low | Low–Medium | Low | **High** |

**Conclusion: C + D combined.** Local features plus an AI rubric judge with deterministic aggregation, with A as the offline fallback. E only as a sanity check.

---

# Fair evaluation design (Step 5)

## Principles

1. **Freeze the rubric and weights *before* scoring the recordings.** Derive them from the feedback text (themes), not by tuning on outputs. Record the frozen version (git tag/hash). Any later change counts as a new version, and its results are reported as "post-hoc".
2. **Blind the inputs.** Copy files to neutral IDs (`R01…R12`) so folder names, batch IDs, dates and speaker names never reach a model. Pseudonymise speakers to `TEACHER` / `STUDENT_n`.
3. **Same inputs to every candidate.** Extract the transcript, features and frames once, cache them, and feed the identical evidence pack to each model and prompt variant. Only the judge changes.
4. **Data handling.** Cloud candidates are run only after the organiser confirms permission. Until then, compare A (rules) and optionally a local model.

## Metrics with the supplied labels (n = 10)

| Metric | Why |
|---|---|
| Exact class accuracy | Required output, but with n = 10 a 95% CI is roughly ±30 points. Report counts (e.g. "7/10"), not percentages alone. |
| Within-one-class accuracy | Ordinal labels: Good vs Excellent confusion is less bad than Poor vs Excellent. |
| Kendall's τ / Spearman ρ vs ordinal label (Poor=0 … Excellent=3) | Does the score at least *order* classes correctly? More robust than accuracy at this n. |
| Confusion matrix | Shows systematic bias (e.g. central tendency). |
| **Criterion-level agreement with feedback** | For each recording, the feedback states or implies ~3–6 criterion facts (homework given? background kept? camera issues? intro adequate?). ≈ 40–50 checkable facts in total, which gives far more signal than 10 labels. Report agreement per criterion. |
| Evidence faithfulness | Fraction of cited timestamps or excerpts that actually exist in the transcript or feature log. Hallucinated evidence is a hard failure. |
| Run-to-run stability | Run each model 3× per recording. Report class flips and score SD. |
| Latency and cost per recording | Practical constraint for the hosted demo. |

## Candidate matrix to compare

| ID | Pipeline | Judge |
|---|---|---|
| R | Rules only (A) | — |
| H-g | Hybrid + rubric (C + D) | `gemma4:31b` |
| H-o | Hybrid + rubric | `gpt-oss:120b` (text only; visual items from local features) |
| H-m | Hybrid + rubric | `glm-5.3-flash` or `minimax-m3.1-flash` |
| H-c | Hybrid + rubric, per-criterion probabilities | `clef` (Decision API) |
| F-g | Free-form (B), as a baseline to show why a rubric is needed | `gemma4:31b` |

Keep the matrix small: ~6 configs × 10 recordings × 3 runs ≈ 180 calls, a few dollars at most. Budget about 45 min of wall time.

## The unlabelled demos and qualitative review

The two demo recordings have **no ground truth**. Use them only to:

- check that the pipeline runs end to end on different conditions (1:1 demo format; a non-English class);
- check that the app **flags reduced reliability** (e.g. "transcript language appears mismatched; conversational criteria less reliable") instead of confidently grading;
- read the explanations for specificity (does each claim cite a timestamp and a concrete behaviour?).

I may record my own impressions, but **they must not be reported as accuracy or ground truth**. They are reviewer notes, clearly labelled as subjective. If two candidates disagree on a demo, report the disagreement; don't adjudicate it as correct or incorrect.

## Guarding against fooling ourselves

- With only 2 examples per class, any configuration that reaches 10/10 after tuning is more likely overfit than good. Prefer the configuration with the best **criterion-level agreement and evidence faithfulness**, not the highest exact accuracy.
- Report all configurations tried, including failures, in the README/explanation.
- Never hardcode anything keyed on file name, batch ID, duration, file size or teacher name. Add a test that renames and re-encodes a sample (e.g. trims 1 minute) and checks the result is essentially unchanged.
