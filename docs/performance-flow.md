# Throughput flow review — 2026-08-19

## Measured bottlenecks removed

1. **Repeated PDF parser setup:** whole-book work kept reopening `pypdf` for
   every chapter. A `PDFDocument` now owns one file descriptor and reader for
   the command, while chapters still render into independent outputs.
2. **Per-chunk file and metadata amplification:** the initial renderer wrote a
   WAV and rewrote a sorted, growing JSON state file for each chunk. It now
   appends 16-bit PCM to one file, then fsyncs one compact JSONL event. Resume
   truncates an unjournaled PCM tail, so the write order is crash-safe.
3. **Cached-model startup network check:** the MLX backend resolves the local
   Hugging Face cache first and provides that directory to Kokoro. Internet is
   used only when weights are not yet installed.
4. **Uncontrolled resource pressure:** CLI profiles now override CPU-side
   thread settings and apply `mlx.set_memory_limit` before model loading.
   Benchmark reports macOS RSS plus MLX active/peak allocation.

## What remains intentionally serial

Kokoro MLX v0.1.2 exposes one serial model stream, not a multi-utterance batch
API. Calling it concurrently would contend for the same accelerator and break
audio ordering. The pipeline instead overlaps bounded CPU cleanup with that
stream and parallelizes final ffmpeg encoding only where the encoder benefits.

## Current M4 Pro evidence

The benchmark uses a 126-second technical sample and tests 500, 900, and
1,400-character chunks. One-pass measurements were:

| Profile | Best chunk | Real-time factor | Process RSS | MLX peak |
| --- | ---: | ---: | ---: | ---: |
| low | 900 | 26.60x | 1,022 MiB | 3,147 MiB |
| balanced | 1400 | 26.83x | 1,022 MiB | 3,147 MiB |
| max | 500 | 23.77x | 1,024 MiB | 3,025 MiB |

This makes `balanced` the current practical default. The difference between
low and balanced is within normal run-to-run variation, whereas max's extra
CPU workers were clearly not useful in this single-pass comparison. Repeat the
benchmark with `--runs 3` on an otherwise idle Mac before changing defaults.

## Full-path validation

A fresh three-page, nine-chunk PDF render produced 176.15 seconds of 24 kHz
AAC/m4b in 11.82 seconds, including model startup, extraction, checkpoints,
and encoding. Its next `--resume` run synthesized zero chunks and returned the
validated output immediately.
