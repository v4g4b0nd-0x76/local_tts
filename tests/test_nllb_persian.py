from types import SimpleNamespace

from local_tts.backends.nllb_persian import _decoder_max_length, _protect_glossary, _restore_glossary


def test_nllb_glossary_terms_are_protected_for_exact_restoration() -> None:
    source, restored = _protect_glossary(
        "An operating system schedules a process; operating systems share the same idea.",
        (("operating system", "سیستم‌عامل"), ("operating systems", "سیستم‌عامل‌ها"), ("process", "فرایند")),
    )

    assert "operating system" not in source
    assert "process" not in source
    assert tuple(target for _, target in restored) == ("سیستم‌عامل‌ها", "سیستم‌عامل", "فرایند")


def test_nllb_glossary_restores_single_bracket_and_bare_token_variants() -> None:
    terms = (("[[LOCAL_TTS_TERM_001]]", "سیستم‌عامل"),)
    assert _restore_glossary("[LOCAL_TTS_TERM_001] LOCAL_TTS_TERM_001", terms) == "سیستم‌عامل سیستم‌عامل"


def test_nllb_generation_uses_only_a_decoder_max_length() -> None:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024))
    assert _decoder_max_length(model, 80) == 81
    assert _decoder_max_length(model, 2048) == 1024
