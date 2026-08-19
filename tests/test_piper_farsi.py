from pathlib import Path

import pytest

from local_tts.backends.piper_farsi import PiperFarsiBackend, piper_voice_path


def test_piper_voice_path_uses_official_download_layout(tmp_path: Path) -> None:
    voice = "fa_IR-ganji_adabi-medium"
    path = tmp_path / "fa" / "fa_IR" / "ganji_adabi" / "medium" / f"{voice}.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"model")
    assert piper_voice_path(tmp_path, voice) == path


def test_piper_voice_path_explains_missing_setup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="setup-farsi"):
        piper_voice_path(tmp_path, "fa_IR-ganji_adabi-medium")


def test_piper_voice_accepts_explicit_stability_controls(tmp_path: Path) -> None:
    backend = PiperFarsiBackend(tmp_path / "voice.onnx", noise_scale=0.45, noise_w_scale=0.65)
    assert backend.noise_scale == 0.45
    assert backend.noise_w_scale == 0.65
    with pytest.raises(ValueError, match="noise_scale"):
        PiperFarsiBackend(tmp_path / "voice.onnx", noise_scale=2.1)
