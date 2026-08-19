from pathlib import Path

from local_tts.backends.kokoro_mlx import KokoroMLXBackend


def test_explicit_local_model_directory_needs_no_hub_lookup(tmp_path: Path) -> None:
    backend = KokoroMLXBackend(model=str(tmp_path))
    assert backend._resolve_model_path() == str(tmp_path)
