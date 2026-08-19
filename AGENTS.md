# local-tts project guidance

## Product intent

`local-tts` is a personal, fully local PDF-to-speech study tool for a Mac mini
M4 Pro with 24 GB unified memory. Python is the v1 implementation language.
It has no cloud service, API key, or paid dependency at run time. Model weights
are downloaded once during explicit local setup and then read from the local
Hugging Face cache or a user-supplied model directory.

## Non-negotiable v1 choices

- Default TTS backend: Kokoro-82M through `kokoro-mlx` on Apple Silicon.
- Keep the backend contract narrow so PyTorch/MPS and CoreML adapters can be
  benchmarked later; do not bake MLX calls into PDF or CLI code.
- Optimise for throughput and controllable host pressure, not broad packaging
  or distribution. The model and its dependencies still retain their licenses.
- Each selected chapter/range writes independent output and chunk checkpoints.
  `--resume` must never redo a completed chunk unless explicitly asked.
- Prefer bounded streaming/prefetching over reading an entire book or its audio
  into memory. The accelerator has one serial inference stream; extraction,
  cleanup, checkpointing, and final encoding may overlap around it.

## Design constraints

- Page numbers in the CLI are human-facing and one-based, inclusive.
- Chapter numbers are one-based positions in the extracted PDF outline.
- Text cleanup must be conservative and configurable: technical prose should
  remain readable; code, tables, URLs, citations, headers/footers, page numbers,
  hyphenated line breaks, and footnotes are separately controllable.
- Keep normal and layout-preserving `pypdf` extraction together. Layout
  classification must be high-confidence: preserve SQL schemas as `schema`
  runs, require multi-row evidence for tables, and exclude dotted contents
  leaders. Preserve PDF metadata in the job manifest and final audio tags.
- Optional conclusions use MLX-LM only behind the summary adapter. Keep source
  context bounded, include only the previous outline chapter plus explicit
  `Chapter N` references, record the exact context in a sidecar, and unload
  the LLM before loading Kokoro so unified-memory footprints never overlap.
- Optional two-voice podcasts use the same source-bound context and a strict
  alternating script. Keep the dialogue model phase separate from Kokoro,
  retain per-turn voice identity in the resume hash, and persist a transcript
  plus model-cost sidecar. The supplied American-English pair is `af_bella`
  (host) and `am_michael` (explainer).
- The optional local HTTP service binds to loopback by default, holds one
  serial Kokoro inference stream, and bounds request size. Its PDF.js bridge
  uses stream chunk source offsets for highlighting; preserve exact source
  offsets even when applying pronunciation rewrites only to spoken text.
- `setup-viewer` installs PDF.js once under `web/node_modules`; `read PDF`
  serves only that selected PDF plus local assets and opens the component
  viewer. Keep PDF load and requested-speech progress visible in the UI.
- Reader pronunciation fixes are explicit configuration mappings applied just
  before synthesis. They must be part of the resume hash and have tests.
- Use ffmpeg only for final audio encoding. Keep a single append-only signed
  16-bit PCM stream plus a durable JSONL checkpoint journal so resume works
  independently of a failed final encode without creating one WAV per chunk.
- Resource profiles (`max`, `balanced`, `low`, `custom`) describe CPU worker
  count, backend batch size, bounded prefetch, chunk size, and a unified-memory
  ceiling. The MLX backend applies that ceiling through `mlx.set_memory_limit`;
  it remains a guideline to MLX, not a limit on all macOS processes.

## Verification

Run `uv run pytest`, `uv run local-tts --help`, and `uv run local-tts benchmark`
after installing model dependencies. A real smoke test must use a real PDF page
range, verify the resulting audio with `ffprobe`, and inspect `--metadata` when
metadata or layout handling changes.
