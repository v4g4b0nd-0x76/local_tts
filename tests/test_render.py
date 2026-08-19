import json
from pathlib import Path

import pytest

import numpy as np

from local_tts.backends.base import TTSResult
from local_tts.models import CleanupOptions, RenderOptions, ResourceSettings
from local_tts.render import render


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str, *, voice: str, speed: float, sample_rate: int) -> TTSResult:
        self.calls += 1
        return TTSResult(np.zeros(240, dtype=np.float32), sample_rate)

    def close(self) -> None:
        pass


def test_render_checkpoints_and_resumes(tmp_path: Path) -> None:
    backend = FakeBackend()
    settings = ResourceSettings(cpu_threads=1, prefetch=1, chunk_chars=100, memory_gb=1)
    options = RenderOptions(audio_format="mp3")
    report = render([(1, "A short sentence for the renderer.")], tmp_path, "page-01", backend, settings, CleanupOptions(), options)
    assert report.output.exists()
    job_dir = tmp_path / ".page-01.local-tts"
    assert (job_dir / "audio.s16le").exists()
    journal = (job_dir / "checkpoints.jsonl").read_text().splitlines()
    assert json.loads(journal[0])["event"] == "chunk"
    assert not (job_dir / "chunks").exists()
    initial_calls = backend.calls
    resumed = render([(1, "A short sentence for the renderer.")], tmp_path, "page-01", backend, settings, CleanupOptions(), RenderOptions(audio_format="mp3", resume=True))
    assert resumed.chunks_synthesized == 0
    assert backend.calls == initial_calls


def test_resume_repairs_unjournaled_audio_tail(tmp_path: Path) -> None:
    backend = FakeBackend()
    settings = ResourceSettings(cpu_threads=1, prefetch=1, chunk_chars=100, memory_gb=1)
    render([(1, "A short sentence for the renderer.")], tmp_path, "page-01", backend, settings, CleanupOptions(), RenderOptions(audio_format="mp3"))
    pcm = tmp_path / ".page-01.local-tts" / "audio.s16le"
    durable_size = pcm.stat().st_size
    with pcm.open("ab") as handle:
        handle.write(b"incomplete tail")
    initial_calls = backend.calls
    render([(1, "A short sentence for the renderer.")], tmp_path, "page-01", backend, settings, CleanupOptions(), RenderOptions(audio_format="mp3", resume=True))
    assert pcm.stat().st_size == durable_size
    assert backend.calls == initial_calls


def test_resume_rejects_changed_source_text(tmp_path: Path) -> None:
    backend = FakeBackend()
    settings = ResourceSettings(cpu_threads=1, prefetch=1, chunk_chars=100, memory_gb=1)
    render([(1, "A short sentence for the renderer.")], tmp_path, "page-01", backend, settings, CleanupOptions(), RenderOptions(audio_format="mp3"))
    with pytest.raises(ValueError, match="does not match"):
        render([(1, "A changed sentence for the renderer.")], tmp_path, "page-01", backend, settings, CleanupOptions(), RenderOptions(audio_format="mp3", resume=True))
