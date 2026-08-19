"""Kokoro-82M backend running on Apple's MLX framework."""

from __future__ import annotations

from pathlib import Path

from .base import TTSResult
from ..models import ResourceSettings

_DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"


class KokoroMLXBackend:
    name = "kokoro-mlx"

    def __init__(self, model: str | None = None) -> None:
        self._model = model
        self._tts = None
        self._memory_limit_bytes: int | None = None

    def configure(self, resources: ResourceSettings) -> None:
        """Apply the profile's real MLX allocation guardrail before loading."""
        try:
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - install-time error
            raise RuntimeError("MLX backend is unavailable; run `uv sync --extra dev`") from exc
        self._memory_limit_bytes = resources.memory_gb * 1024**3
        mx.set_memory_limit(self._memory_limit_bytes)
        # Keep the default cache policy inside the same ceiling. This preserves
        # hot allocations for throughput while MLX reclaims them under pressure.
        mx.set_cache_limit(self._memory_limit_bytes)

    def _load(self):
        if self._tts is None:
            try:
                from kokoro_mlx import KokoroTTS
            except ImportError as exc:  # pragma: no cover - install-time error
                raise RuntimeError("MLX backend is unavailable; run `uv sync --extra dev`") from exc
            self._tts = KokoroTTS.from_pretrained(self._resolve_model_path())
        return self._tts

    def _resolve_model_path(self) -> str:
        """Use cached weights without an HTTP metadata check after first setup."""
        model = self._model or _DEFAULT_MODEL
        if Path(model).is_dir():
            return model
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return snapshot_download(repo_id=model, local_files_only=True)
        except LocalEntryNotFoundError:
            # This only occurs during explicit first use. The resulting path is
            # passed to Kokoro as a directory, so later synthesis is local-only.
            return snapshot_download(repo_id=model)

    def synthesize(self, text: str, *, voice: str, speed: float, sample_rate: int) -> TTSResult:
        result = self._load().generate(text, voice=voice, speed=speed, sample_rate=sample_rate)
        return TTSResult(audio=result.audio, sample_rate=result.sample_rate)

    def close(self) -> None:
        if self._tts is not None:
            self._tts.close()
            self._tts = None

    def memory_stats(self) -> dict[str, float]:
        """Return MLX allocator telemetry in MiB for benchmark reporting."""
        import mlx.core as mx
        return {
            "mlx_active_mb": mx.get_active_memory() / 1024**2,
            "mlx_peak_mb": mx.get_peak_memory() / 1024**2,
            "mlx_cache_mb": mx.get_cache_memory() / 1024**2,
        }

    def reset_memory_peak(self) -> None:
        import mlx.core as mx
        mx.reset_peak_memory()
