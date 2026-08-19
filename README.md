# local-tts

An Apple-Silicon-first, fully local PDF-to-speech study tool. v1 uses
Kokoro-82M with MLX, keeps inference behind a backend interface, and writes
chunk-level checkpoints so interrupted chapter renders resume safely.

## Requirements

- macOS on Apple Silicon
- [uv](https://docs.astral.sh/uv/)
- ffmpeg (`brew install ffmpeg`)
- Python 3.11 or 3.12 (the project intentionally excludes Python 3.13+ until
  the MLX Kokoro dependency supports them)

The first inference downloads the Apache-2.0 Kokoro MLX weights into the local
Hugging Face cache. No API key is needed. Later runs resolve that cache locally
before loading Kokoro, avoiding a model-network or metadata check.

## Setup

```bash
cd ~/projects/local_tts
brew install uv                 # only if uv is not already installed
uv python install 3.12
uv sync --extra dev
uv run pytest
```

## Use

These forms all work (the `book` subcommand is optional):

```bash
uv run local-tts book.pdf --list
uv run local-tts book.pdf --chapter 3
uv run local-tts book.pdf --chapters 3-5 --profile max
uv run local-tts book.pdf --pages 120-160 --format m4b
uv run local-tts book.pdf --pages 120-160 --resume
uv run local-tts book.pdf --config config/study-reader.toml --chapter 3
uv run local-tts benchmark --profile balanced
```

Outputs are chapter/range named, for example `output/chapter-03.m4a`. Durable
intermediate PCM and `checkpoints.jsonl` live below the corresponding hidden
directory, such as `output/.chapter-03.local-tts/`. `--resume` reuses only
chunks whose page, text, voice, speed, and sample-rate hash matches the
checkpoint. Any bytes written before a crash but not checkpointed are safely
truncated before resuming.
`--chapters 3-5` and a whole-book run with an outline produce separate,
independently resumable files for each chapter.

## Resource profiles

| Profile | CPU threads | Batch size | Prefetch | Chunk chars | MLX memory ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| `max` | 8 | 1 | 8 | 1400 | 18 GB |
| `balanced` | 4 | 1 | 4 | 900 | 12 GB |
| `low` | 2 | 1 | 2 | 500 | 6 GB |
| `custom` | explicit flag | explicit flag | explicit flag | explicit flag | explicit flag |

The MLX backend applies the memory value through `mlx.set_memory_limit` before
model load. MLX documents this as an allocation guideline, rather than a hard
limit on the whole macOS process. `--cpu-threads`,
`--batch-size`, `--prefetch`, `--chunk-chars`, and `--memory-gb` override
profile values. The selected Kokoro MLX package exposes a serial inference
instance, so batch size is currently fixed at 1 in practice and retained as a
backend-neutral setting for later adapters.

Run the benchmark on an otherwise idle machine before a long book. It sweeps
500, 900, and 1400-character chunks using a multi-chunk technical passage and
recommends the fastest measurement inside the selected profile's MLX ceiling:

```bash
uv run local-tts benchmark --profile balanced --runs 3
```

## v1 implementation status

- [x] PDF page and extracted-outline inspection; page/chapter/range/whole-book selection
- [x] conservative technical-text cleanup and bounded text chunking
- [x] MLX Kokoro backend; ffmpeg output to m4a, m4b, or mp3
- [x] append-only PCM checkpoints and verified `--resume`
- [x] benchmark sweep with generated-audio-seconds / wall-clock-second, peak RSS, and MLX telemetry
- [ ] compare MLX with official Kokoro PyTorch/MPS and a practical CoreML port
- [ ] richer PDF layout detection for code blocks/tables and embedded-book metadata

## Notes on output and reliability

The model has one serial inference path. v1 keeps it fed by using bounded text
chunks and a bounded cleanup-worker queue while inference runs, and avoids
large accumulated waveform buffers. Final encoding happens from a single
on-disk PCM stream, so an encoding failure does not lose completed synthesis.
The entire command keeps one ordered `pypdf` reader open, avoiding a full PDF
reparse for every independently rendered chapter.

See [technical reading and narration policy](docs/reading-policy.md) for
`skip`/`explain`/`read` handling of code, schemas, and tables plus the supplied
smooth-reader configuration.
