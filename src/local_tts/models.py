"""Stable value objects shared across the CLI and the processing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class ProfileName(StrEnum):
    MAX = "max"
    BALANCED = "balanced"
    LOW = "low"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ResourceSettings:
    """Host-pressure knobs; memory_gb is an advisory unified-memory ceiling."""

    cpu_threads: int
    prefetch: int
    chunk_chars: int
    memory_gb: int
    # Kokoro MLX v0.1.2 generates through one serial model instance. Retain a
    # backend-neutral batch knob for future runtimes, with a truthful default.
    batch_size: int = 1

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


PROFILES: dict[ProfileName, ResourceSettings] = {
    ProfileName.MAX: ResourceSettings(cpu_threads=8, prefetch=8, chunk_chars=1400, memory_gb=18),
    ProfileName.BALANCED: ResourceSettings(cpu_threads=4, prefetch=4, chunk_chars=900, memory_gb=12),
    ProfileName.LOW: ResourceSettings(cpu_threads=2, prefetch=2, chunk_chars=500, memory_gb=6),
}


@dataclass(frozen=True)
class CleanupOptions:
    code: str = "skip"
    schemas: str = "skip"
    tables: str = "skip"
    urls: str = "skip"
    citations: str = "skip"
    footnotes: str = "skip"
    headers_footers: bool = True


@dataclass(frozen=True)
class RenderOptions:
    voice: str = "af_heart"
    speed: float = 1.0
    audio_format: str = "m4a"
    sample_rate: int = 24000
    chunk_pause_ms: int = 0
    pronunciations: tuple[tuple[str, str], ...] = ()
    resume: bool = False
