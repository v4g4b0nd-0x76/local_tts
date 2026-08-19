"""Local Persian narration through the Piper ONNX voices."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import TTSResult
from ..models import ResourceSettings

DEFAULT_FARSI_VOICE = "fa_IR-ganji_adabi-medium"


def default_farsi_model_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "local-tts" / "piper"


def piper_voice_path(model_dir: Path, voice: str) -> Path:
    """Resolve Piper's official download layout without scanning arbitrary paths."""
    try:
        language, name, quality = voice.rsplit("-", 2)
    except ValueError as exc:
        raise ValueError("Farsi Piper voice must look like fa_IR-ganji_adabi-medium") from exc
    expected = model_dir / "fa" / language / name / quality / f"{voice}.onnx"
    fallback = model_dir / f"{voice}.onnx"
    if expected.is_file():
        return expected
    if fallback.is_file():
        return fallback
    raise ValueError(
        f"Persian voice {voice!r} is not installed in {model_dir}; "
        "run `uv run local-tts setup-farsi` first"
    )


class PiperFarsiBackend:
    """Piper backend with Persian phonemization and native Persian voices."""

    name = "piper-farsi"

    def __init__(self, model_path: Path, *, noise_scale: float | None = None, noise_w_scale: float | None = None) -> None:
        if noise_scale is not None and not 0 <= noise_scale <= 2:
            raise ValueError("Piper noise_scale must be between 0 and 2")
        if noise_w_scale is not None and not 0 <= noise_w_scale <= 2:
            raise ValueError("Piper noise_w_scale must be between 0 and 2")
        self.model_path = model_path
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self._voice = None

    def configure(self, resources: ResourceSettings) -> None:
        # Piper's bundled ONNX runtime follows these standard CPU thread limits.
        del resources

    def _load(self):
        if self._voice is None:
            try:
                from piper import PiperVoice
            except ImportError as exc:  # pragma: no cover - install-time concern
                raise RuntimeError("Persian TTS support is unavailable; run `uv sync --extra farsi`") from exc
            self._voice = PiperVoice.load(self.model_path)
        return self._voice

    def synthesize(self, text: str, *, voice: str, speed: float, sample_rate: int) -> TTSResult:
        del voice, sample_rate
        if speed <= 0:
            raise ValueError("reader speed must be greater than zero")
        try:
            from piper import SynthesisConfig
        except ImportError as exc:  # pragma: no cover - _load handles installation too
            raise RuntimeError("Persian TTS support is unavailable; run `uv sync --extra farsi`") from exc
        config = SynthesisConfig(
            length_scale=1 / speed,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
        )
        chunks = list(self._load().synthesize(text, syn_config=config))
        if not chunks:
            return TTSResult(np.array([], dtype=np.float32), 22_050)
        audio = np.concatenate([np.asarray(chunk.audio_float_array, dtype=np.float32) for chunk in chunks])
        return TTSResult(audio, int(chunks[0].sample_rate))

    def close(self) -> None:
        self._voice = None
