"""TTS backend implementations and their small shared contract."""

from .base import TTSBackend, TTSResult
from .kokoro_mlx import KokoroMLXBackend

__all__ = ["KokoroMLXBackend", "TTSBackend", "TTSResult"]

