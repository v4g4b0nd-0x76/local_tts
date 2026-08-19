import numpy as np

from local_tts.backends.base import TTSResult
from local_tts.benchmark import benchmark_candidates
from local_tts.models import ResourceSettings


class FakeBackend:
    name = "fake"

    def synthesize(self, text: str, *, voice: str, speed: float, sample_rate: int) -> TTSResult:
        return TTSResult(np.zeros(max(240, len(text) * 12), dtype=np.float32), sample_rate)

    def close(self) -> None:
        pass


def test_benchmark_probes_unique_chunk_sizes_and_recommends_one() -> None:
    events = []
    reports, recommended = benchmark_candidates(
        FakeBackend(),
        ResourceSettings(cpu_threads=1, prefetch=1, chunk_chars=900, memory_gb=100),
        runs=1,
        progress=events.append,
    )
    assert {report.chunk_chars for report in reports} == {500, 900, 1400}
    assert recommended in reports
    assert all(report.chunks_per_run > 1 for report in reports)
    assert events[-1].completed == events[-1].total == 3
    assert events[-1].audio_seconds > 0
