"""PDF inspection, outline resolution, and one-based page selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from pypdf import PdfReader


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    start_page: int
    end_page: int


class PDFDocument:
    """Keep one PDF reader and file descriptor for an entire CLI operation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: BinaryIO = path.open("rb")
        self.reader = PdfReader(self._stream)

    def inspect(self) -> tuple[int, list[Chapter]]:
        return _inspect_reader(self.reader)

    def extract_pages(self, pages: list[int]) -> list[tuple[int, str]]:
        return [(number, self.reader.pages[number - 1].extract_text() or "") for number in pages]

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "PDFDocument":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def inspect_pdf(path: Path) -> tuple[int, list[Chapter]]:
    with PDFDocument(path) as document:
        return document.inspect()


def _inspect_reader(reader: PdfReader) -> tuple[int, list[Chapter]]:
    starts = _unique_outline_starts(_outline_starts(reader))
    return len(reader.pages), _chapters_from_starts(starts, len(reader.pages))


def _chapters_from_starts(starts: list[tuple[str, int]], page_count: int) -> list[Chapter]:
    chapters: list[Chapter] = []
    for index, (title, start_page) in enumerate(starts, start=1):
        next_start = starts[index][1] if index < len(starts) else page_count + 1
        # Nested PDF bookmarks often point at the same physical page. A
        # render range must never become inverted (for example 7-6).
        end_page = max(start_page, next_start - 1)
        chapters.append(Chapter(index, title, start_page, end_page))
    return chapters


def _outline_starts(reader: PdfReader) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    try:
        outline: Any = reader.outline
    except Exception:
        return result

    def visit(items: list[Any]) -> None:
        for item in items:
            if isinstance(item, list):
                visit(item)
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
                title = str(getattr(item, "title", item)).strip()
            except Exception:
                continue
            if title and (not result or result[-1] != (title, page)):
                result.append((title, page))

    if isinstance(outline, list):
        visit(outline)
    return sorted(result, key=lambda entry: entry[1])


def _unique_outline_starts(starts: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Use one render unit per destination page, retaining its first title."""
    unique: list[tuple[str, int]] = []
    seen_pages: set[int] = set()
    for title, page in starts:
        if page not in seen_pages:
            unique.append((title, page))
            seen_pages.add(page)
    return unique


def page_range(spec: str, page_count: int) -> list[int]:
    """Parse an inclusive 1-based range such as `120-160` or `7`."""
    parts = spec.split("-", maxsplit=1)
    try:
        start = int(parts[0])
        end = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"invalid page range: {spec!r}") from exc
    if start < 1 or end < start or end > page_count:
        raise ValueError(f"page range must lie within 1-{page_count}: {spec!r}")
    return list(range(start, end + 1))


def chapter_pages(chapters: list[Chapter], spec: str) -> tuple[list[int], str]:
    if not chapters:
        raise ValueError("the PDF has no usable outline; select pages with --pages")
    parts = spec.split("-", maxsplit=1)
    try:
        start = int(parts[0])
        end = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"invalid chapter range: {spec!r}") from exc
    if start < 1 or end < start or end > len(chapters):
        raise ValueError(f"chapter range must lie within 1-{len(chapters)}: {spec!r}")
    first, last = chapters[start - 1], chapters[end - 1]
    label = f"chapter-{start:02d}" if start == end else f"chapters-{start:02d}-{end:02d}"
    return list(range(first.start_page, last.end_page + 1)), label


def extract_pages(path: Path, pages: list[int]) -> list[tuple[int, str]]:
    """Compatibility helper; CLI rendering should keep a ``PDFDocument`` open."""
    with PDFDocument(path) as document:
        return document.extract_pages(pages)
