# Runtime selection — 2026-08-19

## Selected v1: `kokoro-mlx==0.1.2`

The default backend is the focused `kokoro-mlx` package, running
Kokoro-82M through MLX on Apple Silicon. It supports Python 3.10–3.12, has no
PyTorch dependency, produces native 24 kHz audio (or optional 48 kHz output),
and lazily obtains the Apache-2.0 MLX weights on first use. Once cached, the
backend resolves the model directory locally and does not perform a Hugging
Face metadata check at each command start.

On this M4 Pro, the 2026-08-19 multi-chunk probe (one measured pass per
candidate) found the balanced profile fastest: 26.83x real time at 1,022 MiB
peak process RSS and 3,147 MiB MLX peak allocation, with 1,400-character
chunks. Low was effectively tied at 26.60x; max was slower at 23.77x, so more
CPU workers did not improve a serial MLX inference stream. These are directional
figures; use `local-tts benchmark --profile balanced --runs 3` before a long
render.

## Deferred benchmark candidates

1. The official `kokoro` Python package provides a PyTorch implementation and
   documents MPS fallback on Apple Silicon. It remains the reference-quality
   candidate for a future `kokoro-mps` adapter.
2. `mlx-audio` supports Kokoro, but is a larger multi-model framework. It was
   not chosen for v1 because `kokoro-mlx` has a smaller focused interface.
3. A CoreML route is not selected yet. Available ports are currently centered
   on Swift/iOS/macOS rather than a maintained Python interface that fits this
   project. Add one only after a reproducible CoreML Python integration can
   beat MLX throughput or materially reduce host pressure on this M4 Pro.

The `TTSBackend` protocol isolates this decision. Benchmarks must report
generated audio seconds, wall time, real-time factor, profile, chunk size,
batch setting, system peak RSS, and MLX allocator telemetry before replacing
MLX as the default.
