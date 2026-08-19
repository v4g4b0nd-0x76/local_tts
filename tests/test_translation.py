from pathlib import Path

from local_tts.models import CleanupOptions, ResourceSettings
from local_tts.pdf import ExtractedPage, LayoutStats
from local_tts.summarize import SummaryResult
from local_tts.text import clean_page
from local_tts.translation import TranslationOptions, apply_transliterations, normalize_persian, translate_pages_to_persian


class FakePersianTranslator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, tuple[tuple[str, str], ...]]] = []

    def translate_to_persian(
        self,
        source_text: str,
        *,
        max_tokens: int,
        glossary: tuple[tuple[str, str], ...] = (),
    ) -> SummaryResult:
        self.calls.append((source_text, max_tokens, glossary))
        return SummaryResult(
            text="يك كتاب درباره كرنل و فرايندها.",
            model="fake-qwen",
            input_tokens=8,
            output_tokens=9,
            wall_seconds=0.1,
            mlx_active_mb=12.0,
            mlx_peak_mb=16.0,
        )


def _page(text: str) -> ExtractedPage:
    return ExtractedPage(1, text, {}, LayoutStats(0, 0, 0))


def test_normalize_persian_uses_persian_letters_and_removes_tatweel() -> None:
    assert normalize_persian("علي كـتاب و يکى") == "علی کتاب و یکی"


def test_transliterations_use_exact_user_approved_persian_names() -> None:
    assert apply_transliterations("Hennessy and Patterson", (("Hennessy", "هنسی"), ("Patterson", "پترسون"))) == "هنسی and پترسون"


def test_cleanup_drops_roman_page_numbers() -> None:
    assert clean_page("vi\nUseful study text.", CleanupOptions()) == "Useful study text."


def test_translation_is_durable_reviewable_and_resumable(tmp_path: Path) -> None:
    translator = FakePersianTranslator()
    resources = ResourceSettings(cpu_threads=1, prefetch=1, chunk_chars=900, memory_gb=1)
    options = TranslationOptions(
        model="fake-qwen",
        backend="qwen",
        max_source_chars=900,
        max_output_tokens=128,
        glossary=(("kernel", "هسته"),),
    )
    first = translate_pages_to_persian(
        [_page("A kernel controls a process.")],
        tmp_path,
        "pages-1-fa",
        translator,
        resources,
        CleanupOptions(headers_footers=False),
        options,
        resume=False,
    )

    assert first.sidecar.name == "pages-1-fa.translation.json"
    assert first.segments[0].translation == "یک کتاب درباره کرنل و فرایندها."
    assert len(translator.calls) == 1
    assert (tmp_path / ".pages-1-fa.local-tts" / "translation.jsonl").is_file()
    assert '"source"' in first.sidecar.read_text()

    resumed = translate_pages_to_persian(
        [_page("A kernel controls a process.")],
        tmp_path,
        "pages-1-fa",
        translator,
        resources,
        CleanupOptions(headers_footers=False),
        options,
        resume=True,
    )
    assert resumed.segments == first.segments
    assert len(translator.calls) == 1


def test_protected_terms_override_glossary_and_transliteration(tmp_path: Path) -> None:
    translator = FakePersianTranslator()
    resources = ResourceSettings(cpu_threads=1, prefetch=1, chunk_chars=900, memory_gb=1)
    options = TranslationOptions(
        model="fake-qwen",
        backend="qwen",
        max_source_chars=900,
        max_output_tokens=128,
        protected_terms=("kernel", "Hennessy"),
        glossary=(("kernel", "هسته"), ("process", "فرایند")),
        transliterations=(("Hennessy", "هنسی"),),
    )

    translate_pages_to_persian(
        [_page("Hennessy explains a kernel process.")],
        tmp_path,
        "protected-fa",
        translator,
        resources,
        CleanupOptions(headers_footers=False),
        options,
        resume=False,
    )

    assert translator.calls[0][2] == (("kernel", "kernel"), ("Hennessy", "Hennessy"), ("process", "فرایند"))


def test_translation_emits_page_progress(tmp_path: Path) -> None:
    translator = FakePersianTranslator()
    events = []
    report = translate_pages_to_persian(
        [_page("A kernel controls a process.")],
        tmp_path,
        "progress-fa",
        translator,
        ResourceSettings(cpu_threads=1, prefetch=1, chunk_chars=900, memory_gb=1),
        CleanupOptions(headers_footers=False),
        TranslationOptions(model="fake-qwen", backend="qwen", max_source_chars=900, max_output_tokens=128),
        resume=False,
        progress=events.append,
    )

    assert report.segments
    assert events[-1].phase == "writing translation"
    assert events[-1].completed == events[-1].total == 1
