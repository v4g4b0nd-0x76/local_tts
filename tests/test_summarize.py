import json
from pathlib import Path

from local_tts.models import CleanupOptions
from local_tts.pdf import Chapter, ExtractedPage, LayoutStats
from local_tts.summarize import SummaryOptions, SummaryResult, build_summary_context, write_summary


class FakeDocument:
    def __init__(self, pages: dict[int, ExtractedPage]) -> None:
        self.pages = pages

    def extract_pages(self, numbers: list[int]) -> list[ExtractedPage]:
        return [self.pages[number] for number in numbers]


def _page(number: int, text: str) -> ExtractedPage:
    return ExtractedPage(number, text, {}, LayoutStats(0, 0, 0))


def test_summary_context_includes_previous_and_explicitly_referenced_chapters() -> None:
    pages = {
        1: _page(1, "Earlier chapter explains a process abstraction."),
        2: _page(2, "It establishes the baseline."),
        3: _page(3, "This section applies the baseline. See Chapter 3 for the scheduler details."),
        4: _page(4, "The mechanism produces a useful result."),
        5: _page(5, "The referenced scheduler chapter defines tickets."),
        6: _page(6, "Tickets determine proportional share."),
    }
    chapters = [
        Chapter(1, "Baseline", 1, 2),
        Chapter(2, "Application", 3, 4),
        Chapter(3, "Scheduling", 5, 6),
    ]
    context = build_summary_context(
        FakeDocument(pages),  # type: ignore[arg-type]
        [pages[3], pages[4]],
        chapters,
        CleanupOptions(code="read", schemas="read", tables="read"),
        SummaryOptions(max_context_chars=10_000, max_reference_chapters=3),
    )

    assert context.previous_chapter == chapters[0]
    assert context.referenced_chapters == (chapters[2],)
    assert [source.role for source in context.sources] == ["selected pages", "previous chapter", "referenced chapter"]
    assert "scheduler details" in context.prompt


def test_summary_sidecars_are_inspectable(tmp_path: Path) -> None:
    pages = {1: _page(1, "A selected concept.")}
    context = build_summary_context(
        FakeDocument(pages),  # type: ignore[arg-type]
        [pages[1]],
        [],
        CleanupOptions(),
        SummaryOptions(max_context_chars=4_000),
    )
    result = SummaryResult("The concept has one clear takeaway.", "local/model", 100, 12, 1.2, 250.0, 300.0)
    markdown, details = write_summary(tmp_path, "pages-1", context, result)

    assert "one clear takeaway" in markdown.read_text()
    assert json.loads(details.read_text())["result"]["input_tokens"] == 100
