from pathlib import Path

from local_tts.cli import _apply_book_config, _apply_serve_config, _parser


def test_book_config_applies_smooth_reader_and_explain_policies(tmp_path: Path) -> None:
    config = tmp_path / "reader.toml"
    config.write_text(
        """[reader]
voice = "bf_emma"
speed = 0.92
sample_rate = 24000

[summary]
model = "local/model"
context_chars = 9000
max_output_tokens = 320
max_references = 2

[podcast]
host_voice = "af_bella"
explainer_voice = "am_michael"
max_output_tokens = 400
max_turns = 6
turn_pause_ms = 450

[pronunciation]
"prob lem" = "problum"

[cleanup]
code = "explain"
schemas = "explain"
tables = "skip"
headers_footers = false

[resources]
profile = "balanced"

[output]
format = "m4b"
directory = "spoken"
"""
    )
    args = _parser().parse_args(["book", "book.pdf", "--config", str(config)])
    _apply_book_config(args)
    assert args.voice == "bf_emma"
    assert args.speed == 0.92
    assert args.code == args.schemas == "explain"
    assert args.tables == "skip"
    assert args.keep_headers_footers is True
    assert args.format == "m4b"
    assert args.output_dir == Path("spoken")
    assert args.summary_model == "local/model"
    assert args.summary_context_chars == 9000
    assert args.summary_max_tokens == 320
    assert args.summary_references == 2
    assert args.podcast_host_voice == "af_bella"
    assert args.podcast_explainer_voice == "am_michael"
    assert args.podcast_max_tokens == 400
    assert args.podcast_max_turns == 6
    assert args.podcast_turn_pause_ms == 450
    assert args.pronunciations == (("prob lem", "problum"),)


def test_explicit_cli_reader_value_beats_config(tmp_path: Path) -> None:
    config = tmp_path / "reader.toml"
    config.write_text("[reader]\nvoice = \"bf_emma\"\nspeed = 0.92\n")
    args = _parser().parse_args(["book", "book.pdf", "--config", str(config), "--voice", "af_bella", "--speed", "0.96"])
    _apply_book_config(args)
    assert args.voice == "af_bella"
    assert args.speed == 0.96


def test_misspelled_summerize_alias_is_supported() -> None:
    args = _parser().parse_args(["book", "book.pdf", "--summerize"])
    assert args.summarize is True


def test_server_config_reuses_reader_and_resource_settings(tmp_path: Path) -> None:
    config = tmp_path / "reader.toml"
    config.write_text(
        """[reader]
voice = "af_bella"
speed = 0.94
sample_rate = 24000
chunk_pause_ms = 120

[resources]
profile = "low"
"""
    )
    args = _parser().parse_args(["serve", "--config", str(config), "--port", "9876"])
    _apply_serve_config(args)

    assert args.port == 9876
    assert args.voice == "af_bella"
    assert args.speed == 0.94
    assert args.chunk_pause_ms == 120
    assert args.profile == "low"


def test_read_command_reuses_server_config_and_can_skip_browser_open(tmp_path: Path) -> None:
    config = tmp_path / "reader.toml"
    config.write_text("[reader]\nvoice = \"af_bella\"\n")
    args = _parser().parse_args(["read", "book.pdf", "--config", str(config), "--no-open", "--port", "8765"])
    _apply_serve_config(args)

    assert args.pdf == Path("book.pdf")
    assert args.no_open is True
    assert args.voice == "af_bella"
