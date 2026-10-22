# Ollama Cloud model comparison

Access date for all sources: **2026-10-22**. Sources:

- Cloud model catalogue: <https://ollama.com/search?c=cloud>
- Live model list with capability flags: <https://ollama.com/api/tags>
- Pricing, plans, concurrency and privacy FAQ: <https://ollama.com/pricing>
- Cloud docs (API key, endpoint, data handling): <https://docs.ollama.com/cloud>
- Vision: <https://docs.ollama.com/capabilities/vision.md>
- Chat API: <https://docs.ollama.com/api/chat.md>
- Structured outputs: <https://docs.ollama.com/capabilities/structured-outputs.md>
- Decision / System One: <https://docs.ollama.com/capabilities/decision.md>, <https://docs.ollama.com/api/systemone.md>
- Gemma 4 model card: <https://ollama.com/library/gemma4>
- Pricing blog (plan changes, 2026-08-31): <https://ollama.com/blog/transparent-pricing>

No Ollama API was called during this research. Benchmark and quality claims below are **vendor claims**; none were tested on this dataset.

## 1. Headline findings

1. **No Ollama Cloud model accepts video.** The chat API's only media field is `images` (base64 stills). The System One API states explicitly that "streaming, **video**, tools, and generation controls are not supported." Every cloud workflow must therefore **extract frames and transcribe audio locally first**.
2. **No Ollama Cloud model accepts audio.** `/api/tags` lists capabilities `completion`, `vision`, `thinking`, `tools`, `decision`; none lists audio. The catalogue tags the Gemma 4 *family* with "audio", but the model card says only the **E2B/E4B** edge variants support audio. The **cloud variant is `gemma4:31b`, which is "Text, Image"** with "No Audio". Even the local E2B/E4B tags on Ollama are listed as "Text, Image", so audio input via Ollama is **unverified**.
3. **Schema-enforced JSON is not available on cloud.** The Structured Outputs page says: *"Ollama's cloud currently does not support structured outputs."* JSON must be requested in the prompt, then validated (Pydantic) and retried on failure. The `format` parameter only constrains output on local models.
4. **The Decision (System One) API *does* give structured, probabilistic answers.** The `clef` model (27B, vision, 64K context) answers typed questions (`choice`, yes/no `noul`, ordered `score` with 2–10 levels) and returns probabilities and a confidence value. The cloud version takes up to 4 images (≤ 4 MiB each, 8 MiB total), up to 64 questions, and a 65,536-token input. **This is a good fit for rubric scoring**, but it produces no free-text explanation.
5. **Data handling (vendor statement):** prompts and responses are "never logged or trained on". Partner hosts must have "no logging, no training, and zero data retention". Hosting is "primarily in the United States", with possible routing to Europe. **This does not equal organiser permission.** The challenge says to process recordings "only using services specifically permitted by the organiser", and no service is named. **Confirm before sending anything, even derived data.**

## 2. Candidate models

Pricing per 1M tokens (standard rate). Context and capabilities come from the catalogue and `/api/tags`.

| Model (API name) | Vision | Thinking | Context | Input $ | Output $ | Notes for this task |
|---|---|---|---|---|---|---|
| `gemma4:31b` | ✅ | ✅ | 256K | 0.14 | 0.40 | Cheapest capable VLM. Variable image token budget (70–1120 tokens/image) suits many low-res frames. Vendor benchmark MMMU-Pro 76.9%. "Low usage" tier. |
| `glm-5.3-flash` | ✅ | ✅ | 1M | 0.15 | 0.50 | Cheap, huge context, natively multimodal (vendor claim). |
| `minimax-m3.1-flash` | ✅ | ✅ | 976K | 0.15 | 0.60 | Cheap, large context. Newest (2026-10-19), so less field history. |
| `deepseek-v4.1-flash` | ✅ | ✅ | 1M | 0.30 | 1.20 | Off-peak half price. |
| `mistral-large-4` | ✅ | ✅ | 1M | 0.68 | 2.09 | 1.6B vision encoder. Mid-price. |
| `muse-spark-open` | ✅ | ✅ | 1M | 1.25 | 4.25 | Large, "High usage". |
| `kimi-k2.6` / `kimi-k2.7-code` | ✅ | ✅ | 256K | 0.95 | 4.00 | Coding/agentic focus. |
| `kimi-k3` | ✅ | ✅ | 1M | 3.00 | 15.00 | Most expensive. Not justified here. |
| `gpt-oss:120b` | ❌ | ✅ | — | 0.15 | 0.60 | Text-only judge over transcript + features. |
| `nemotron-3-super` | ❌ | ✅ | 256K | 0.015 | 0.60 | Text-only, very cheap input. |
| `glm-5.3`, `deepseek-v4-pro`, `nemotron-3-ultra` | ❌ | ✅ | 256K–1M | 0.10–1.40 | 3.00–4.40 | Text-only reasoning; no advantage over cheaper options for this task. |
| `beam` | ❌ | ✅ | 1M | **free** | **free** | Text-only. Free listing per catalogue; free-plan limits **unverified**. |
| `clef` (`clef:27b`) | ✅ (≤4 imgs) | — | 64K | 0.24 | — | **Decision API only.** Calibrated choice/score answers; no prose. |

### Educational reasoning

No vendor publishes an education-specific evaluation. Reasoning about "is there a rapport-building intro, are students included, is homework given" is a **reading-comprehension and rubric-following** task over a ~5–10k-word transcript. Any of the general-purpose models above should handle it; which one is more faithful to evidence is **unverified** and must be tested (see `approach_comparison.md`, evaluation section).

## 3. Direct video vs extract-first workflow

| | Direct video upload | Extract-first workflow (only option on Ollama Cloud) |
|---|---|---|
| Supported on Ollama Cloud? | **No** (no video field; System One explicitly excludes video) | Yes |
| What is sent | — | Text (pseudonymised transcript, numeric features) plus a few still images (≤ 4 for Clef; chat-API image count limit **unverified**) |
| Temporal understanding | — | Only what we encode: timestamps in the transcript, per-frame captions, feature timelines |
| Privacy | — | Better: we choose exactly what leaves the machine and can redact names and blur faces first |
| Cost | — | Tiny (see below) |

## 4. Usage limits and cost

- **Plans** (pricing page): Free ($0, "starter usage credits" for "a smaller set of starter models", 1 concurrent request); Pro $20/mo with $60 credits (3 concurrent); Max $100/mo with $300 credits (10 concurrent); Team $500/mo. Off-peak discounts apply outside 12:00–18:00 UTC on weekdays and all weekend (for models with an off-peak row).
- **Which models the Free plan includes is not published**: "starter models" are not named (**unverified**). The `clef` cloud model requires "a paid plan or purchased usage credits".
- **Per-recording cost estimate** (my arithmetic, not measured). A 50-min class transcript ≈ 5–8k words ≈ 8–12k tokens. Add features and prompt ≈ 15k tokens, plus ~8 frames at a few hundred tokens each (cloud image token accounting **unverified**), ≈ 20k input and ~1.5k output tokens.
  - `gemma4:31b`: ≈ 20k × $0.14/M + 1.5k × $0.40/M ≈ **$0.003**
  - `kimi-k3`: ≈ $0.06 + $0.02 ≈ **$0.08**
  - `clef` (8 questions × 4 images, shared state counted per question): ≈ 100–300k input tokens ≈ **$0.03–0.07** (estimate; billing counts shared context per question)
- Cost is therefore not a constraint. **Availability, permission and latency are.**

## 5. Local Ollama alternative

- The machine has the Ollama CLI 0.40.8, but the **local server is 0.21.2** (the CLI warned of a version mismatch). Cloud models via the local server and `clef:cloud` need server ≥ 0.40.5. Calling `https://ollama.com/api/...` directly with an API key avoids this.
- Running `gemma4:e4b` or `gemma4:12b` locally (no data leaves the machine) is feasible on 32 GB RAM, but slow on CPU-only hardware, and not feasible on a free hosting tier. It is a reasonable **privacy fallback for offline evaluation**, not for the hosted app.

## 6. Shortlist

| Role | First choice | Backup | Why |
|---|---|---|---|
| Transcript + evidence judge (prose explanation) | `gemma4:31b` | `gpt-oss:120b` or `glm-5.3-flash` | Cheap, vision-capable (one model covers frames too), 256K context covers a full class |
| Visual yes/no checks (background, teacher visible, lighting) | `gemma4:31b` (chat, prompt-JSON) | `clef` (Decision API) | Clef returns probabilities, which suit thresholds |
| Ordinal rubric scoring with probabilities | `clef` | — | Native `score` type over ordered levels |

All of the above is **conditional on organiser permission** to send pseudonymised transcripts and (face-blurred) frames to Ollama Cloud.

## 7. Unverified items to check before building

- Free-plan "starter models" list, and whether `gemma4:31b` is included.
- Maximum images per `/api/chat` request on cloud, and per-image token billing.
- Whether `think: false` / `think: "low"` reduces latency on `gemma4:31b` cloud.
- Real-world rate limits on the Free plan (only concurrency is documented).
- Whether the organiser permits Ollama Cloud at all.
