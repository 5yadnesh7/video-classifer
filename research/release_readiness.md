# Release readiness

Date: **2026-10-22**. Author: deployment and release pass. Scope: documentation only.

## Status

Cloud permission is **pending**. The organiser has not approved Ollama Cloud, or any other off-machine processor, for these class recordings or for data derived from them.

No live Ollama call has been made by this task. `OLLAMA_CLOUD_PERMITTED` was not set. The application does not set it. A value other than exactly `1` leaves cloud scoring off, including the value `true`.

Hosted operation is **not verified**. Nothing has been deployed. No public server was started. No Space was created. `share=True` was not enabled. No load test was run. No recording was uploaded anywhere.

The supported way to run the app today is rules-only, on the local machine. See `README.md`.

After the five specialist passes were merged, the project suite was `105 passed`, `0 failed`, `0 skipped` on Python 3.12. That run used synthetic tests only. It did not call Ollama and did not open the supplied recordings.

## What is ready

- Python 3.12 local setup is documented: `py -3.12 -m venv .venv`, then install from `requirements.txt` with that virtualenv's pip. `ffmpeg` and `ffprobe` must be on `PATH`.
- Rules-only entry points are documented: `.venv\Scripts\python.exe -m class_quality.app` (binds to `127.0.0.1`) and `.venv\Scripts\python.exe -m class_quality --video ... --vtt ...`.
- `.env` is gitignored. `.env.example` names `OLLAMA_CLOUD_PERMITTED` and `OLLAMA_API_KEY` and states that only the exact value `1` can enable cloud scoring.
- Recordings, transcripts, and `.env` are gitignored.
- The rules-only baseline stands as an aggregate: **5/10 exact**, **9/10 within one adjacent class**. Weights were not retuned.

## Blockers

1. Organiser permission for any cloud processing, including pseudonymised text and face-blurred frames. Until that approval, keep `OLLAMA_CLOUD_PERMITTED` unset and do not call Ollama.
2. Children's class recordings must not be uploaded to a public Space or a Gradio `share=True` URL. Local-only hosting is the recommendation until the organiser approves cloud processing.
3. A free Gradio Space is not a verified place to process one 40–90 minute 1080p recording. CPU Basic is 2 vCPU, 16 GB RAM, and 50 GB of disk that does not persist, and creating a Gradio Space on compute requires a paid plan. A free personal account's Gradio exception is ZeroGPU (at most two Spaces), whose documented GPU function default is 60 seconds and whose free daily GPU quota is 5 minutes. Those GPU figures apply to `@spaces.GPU`. This app does not use that decorator. See the verified and unverified lists below.
4. Wall-clock time and memory for one 40–90 minute 1080p recording on this CPU-only Windows machine were not measured.
5. Name matching now ignores case. Tokens shorter than 3 characters are still not removed. Stills are not attached on the cloud path, because they are not face-blurred. Visual criteria stay not assessed. `launch` passes `share=False`.

## Hosting limits verified on 2026-10-22

| Topic | Verified statement | URL |
|---|---|---|
| Gradio upload size | `max_file_size=None` means no library limit. This app's `launch` call does not set the parameter. | https://www.gradio.app/docs/gradio/blocks |
| Gradio public share | `share=True` publishes a public URL. Share links expire after one week. The share servers proxy the local process. | https://gradio.app/guides/sharing-your-app |
| Gradio long jobs | The default queue uses server-sent events so a job is not subject to the roughly one-minute browser POST timeout described in the guide. `time_limit` is documented as applying to `.stream()`. | https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance and https://www.gradio.app/docs/gradio/on |
| Space CPU and memory | Default environment: 16 GB RAM, 2 CPU cores, 50 GB of disk that is not persistent. CPU Basic row: 2 vCPU, 16 GB, hourly price free. A Gradio or Docker Space on compute requires a paid plan. Static Spaces are free. Free personal accounts in good standing can host up to 2 ZeroGPU Gradio Spaces. | https://huggingface.co/docs/hub/spaces-overview |
| Space disk lifetime | That disk is ephemeral. Contents are lost if the Space restarts or stops. | https://huggingface.co/docs/hub/spaces-storage |
| Old persistent-storage flag | `suggested_storage` is ignored. The persistent storage feature is no longer available. Buckets can be attached as volumes. | https://huggingface.co/docs/hub/spaces-config-reference and https://huggingface.co/docs/hub/spaces-storage |
| Sleep | Free `cpu-basic` hardware sleeps after 48 hours of inactivity. Custom sleep time cannot be set on `cpu-basic`. | https://huggingface.co/docs/huggingface_hub/package_reference/space_runtime and https://huggingface.co/docs/huggingface_hub/guides/manage-spaces |
| Startup timeout | Default `startup_duration_timeout` is 30 minutes. This limits startup, not one analysis request. | https://huggingface.co/docs/hub/spaces-config-reference |
| ZeroGPU execution quota | Default `@spaces.GPU` duration is 60 seconds. Free-account daily GPU quota is 5 minutes. Unauthenticated quota is 2 minutes. | https://huggingface.co/docs/hub/spaces-zerogpu |
| Space visibility | A public Space is open to everyone. A protected Space still serves the running app on its embed URL. | https://huggingface.co/docs/hub/spaces-overview |
| `ffmpeg` image | The default Gradio Spaces image includes `ffmpeg`. | https://huggingface.co/docs/hub/spaces-dependencies |

## Left unverified on purpose

- The maximum size of a file uploaded through a running Space's Gradio widget. Hub repository limits (including the recommendation to split files under 200 GB) are a different limit and were not treated as the widget limit. https://huggingface.co/docs/hub/storage-limits
- A per-request execution timeout for CPU work on a Gradio Space. The 30-minute figure above is startup. The 60-second ZeroGPU figure is for `@spaces.GPU`, which this app does not use.
- The maximum duration argument `@spaces.GPU` accepts. The page shows a 60-second default and a 120-second example and does not state a maximum.
- Whether undecorated CPU work on a ZeroGPU Space counts against the GPU duration or the 5-minute free daily quota.
- The numeric free allowance for Storage Buckets. The buckets page says a free allowance exists and points at pricing; the gigabyte figure was not on that page. https://huggingface.co/docs/hub/storage-buckets
- Byte size of one 40–90 minute 1080p class recording, and whether that file plus a temporary copy fits in 50 GB.
- Wall-clock time and peak memory of the rules-only `ffmpeg` measurement on this CPU-only Windows machine.
- This machine's RAM and CPU count. An earlier research note describes the machine; this task did not remeasure it.
- Any claim that this app has been deployed, shared, or load-tested.

## Weights

Do not retune rubric weights or cutoffs from the 10-recording baseline. The evaluation note already says the scored rules miss tone, camera presence, and setup, which is why the lower labels are weak. Those criteria stay unassessed until cloud processing is permitted.
