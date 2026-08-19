"""Backend contract kept independent from document processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..models import ResourceSettings


@dataclass(frozen=True)
class TTSResult:
    audio: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.audio) / self.sample_rate


class TTSBackend(Protocol):
    name: str

    def configure(self, resources: ResourceSettings) -> None: ...

    def synthesize(self, text: str, *, voice: str, speed: float, sample_rate: int) -> TTSResult: ...

    def close(self) -> None: ...
