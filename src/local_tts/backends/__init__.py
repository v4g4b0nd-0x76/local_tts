"""TTS backend implementations and their small shared contract."""

from .base import TTSBackend, TTSResult
from .kokoro_mlx import KokoroMLXBackend
from .nllb_persian import NLLBPersianTranslator
from .piper_farsi import PiperFarsiBackend

__all__ = ["KokoroMLXBackend", "NLLBPersianTranslator", "PiperFarsiBackend", "TTSBackend", "TTSResult"]
