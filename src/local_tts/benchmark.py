"""Repeatable throughput probes for the selected local TTS backend."""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import asdict, dataclass, replace

from .backends.base import TTSBackend
from .models import ResourceSettings
from .text import chunk_text

# Long enough to cross every default chunk size and expose per-call overhead,
# while remaining short enough for an interactive benchmark run.
_SAMPLE = " ".join([
    "This local benchmark uses technical prose rather than a single short sentence.",
    "A useful result measures generated audio duration against wall clock time and records process memory pressure.",
    "The renderer extracts PDF text, performs conservative cleanup, splits at sentence boundaries, and synthesizes each durable chunk locally.",
    "Chunk size controls checkpoint frequency and Python to model call overhead, while the current Kokoro MLX backend retains one serial inference stream.",
    "The benchmark therefore varies chunk size around the selected profile and recommends the fastest result that remains under its advisory unified memory ceiling.",
] * 3)


@dataclass(frozen=True)
class BenchmarkReport:
    chunk_chars: int
    chunks_per_run: int
    runs: int
    audio_seconds: float
    wall_seconds: float
    realtime_factor: float
    peak_rss_mb: float
    accelerator_active_mb: float | None
    accelerator_peak_mb: float | None
    within_memory_ceiling: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark(backend: TTSBackend, resources: ResourceSettings, runs: int = 3) -> BenchmarkReport:
    """Measure a representative multi-chunk passage after one model warm-up."""
    chunks = chunk_text(_SAMPLE, resources.chunk_chars)
    backend.synthesize(chunks[0], voice="af_heart", speed=1.0, sample_rate=24000)
    started = time.perf_counter()
    audio_seconds = 0.0
    for _ in range(runs):
        for text in chunks:
            result = backend.synthesize(text, voice="af_heart", speed=1.0, sample_rate=24000)
            audio_seconds += result.duration_seconds
    elapsed = time.perf_counter() - started
    rtf = audio_seconds / elapsed if elapsed else 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    telemetry = getattr(backend, "memory_stats", lambda: {})()
    return BenchmarkReport(
        chunk_chars=resources.chunk_chars,
        chunks_per_run=len(chunks),
        runs=runs,
        audio_seconds=audio_seconds,
        wall_seconds=elapsed,
        realtime_factor=rtf,
        peak_rss_mb=peak_mb,
        accelerator_active_mb=telemetry.get("mlx_active_mb"),
        accelerator_peak_mb=telemetry.get("mlx_peak_mb"),
        within_memory_ceiling=peak_mb <= resources.memory_gb * 1024,
    )


def benchmark_candidates(
    backend: TTSBackend,
    resources: ResourceSettings,
    runs: int = 3,
) -> tuple[list[BenchmarkReport], BenchmarkReport]:
    """Probe practical chunk sizes and return the best memory-safe result."""
    candidates = tuple(dict.fromkeys((500, 900, 1400, resources.chunk_chars)))
    reports = [benchmark(backend, replace(resources, chunk_chars=size), runs=runs) for size in candidates]
    safe = [report for report in reports if report.within_memory_ceiling]
    if not safe:
        raise RuntimeError("every benchmark candidate exceeded the advisory memory ceiling")
    return reports, max(safe, key=lambda report: report.realtime_factor)
