"""PDF inspection, outline resolution, and one-based page selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

from pypdf import PdfReader


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    start_page: int
    end_page: int


@dataclass(frozen=True)
class ExtractedPage:
    """Plain text plus layout-derived classifications for a source page."""

    number: int
    text: str
    technical_lines: dict[str, str]
    layout_stats: LayoutStats


@dataclass(frozen=True)
class LayoutStats:
    code_lines: int
    schema_lines: int
    table_lines: int

    def as_dict(self) -> dict[str, int]:
        return {
            "code_lines": self.code_lines,
            "schema_lines": self.schema_lines,
            "table_lines": self.table_lines,
        }


class PDFDocument:
    """Keep one PDF reader and file descriptor for an entire CLI operation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._stream: BinaryIO = self.path.open("rb")
        self.reader = PdfReader(self._stream)

    def inspect(self) -> tuple[int, list[Chapter]]:
        return _inspect_reader(self.reader)

    def extract_pages(
        self,
        pages: list[int],
        *,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> list[ExtractedPage]:
        """Extract one selected range while exposing page-granular I/O progress."""
        extracted: list[ExtractedPage] = []
        total = len(pages)
        for index, number in enumerate(pages, start=1):
            extracted.append(self._extract_page(number))
            if progress is not None:
                progress(number, index, total)
        return extracted

    def metadata(self) -> dict[str, object]:
        """Read embedded document metadata without requiring external tools."""
        raw = self.reader.metadata or {}

        def value(key: str) -> str:
            item = raw.get(key)
            return str(item).strip() if item else ""

        try:
            attachments = sorted(self.reader.attachments.keys())
        except Exception:
            attachments = []
        return {
            "source_file": self.path.name,
            "title": value("/Title") or self.path.stem,
            "author": value("/Author"),
            "subject": value("/Subject"),
            "keywords": value("/Keywords"),
            "creator": value("/Creator"),
            "producer": value("/Producer"),
            "creation_date": value("/CreationDate"),
            "modification_date": value("/ModDate"),
            "page_count": len(self.reader.pages),
            "encrypted": self.reader.is_encrypted,
            "attachments": attachments,
            "toc_entries": len(_unique_outline_starts(_outline_starts(self.reader))),
        }

    def _extract_page(self, number: int) -> ExtractedPage:
        page = self.reader.pages[number - 1]
        plain = page.extract_text() or ""
        # pypdf's layout mode retains enough horizontal structure to identify
        # code and tabular runs without a second native PDF engine.
        try:
            layout = page.extract_text(extraction_mode="layout") or plain
        except (KeyError, ValueError):
            # Some valid pages (notably deliberately blank pages) have no
            # content stream for pypdf's layout extractor to inspect.
            layout = plain
        technical_lines = _classify_layout_lines(layout)
        stats = LayoutStats(
            code_lines=sum(kind == "code" for kind in technical_lines.values()),
            schema_lines=sum(kind == "schema" for kind in technical_lines.values()),
            table_lines=sum(kind == "table" for kind in technical_lines.values()),
        )
        # The ordinary extractor often emits visual character spacing as broken
        # English words (for example ``hono r``); layout mode is usually much
        # cleaner for book prose. Select it only when it retained comparable
        # content and measurably reduces those split-word artifacts.
        return ExtractedPage(number, _preferred_reading_text(plain, layout), technical_lines, stats)

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


def extract_pages(path: Path, pages: list[int]) -> list[ExtractedPage]:
    """Compatibility helper; CLI rendering should keep a ``PDFDocument`` open."""
    with PDFDocument(path) as document:
        return document.extract_pages(pages)


def _classify_layout_lines(layout: str) -> dict[str, str]:
    """Classify high-confidence technical runs from layout-preserving text."""
    lines = layout.splitlines()
    classifications: dict[str, str] = {}
    active_kind: str | None = None
    table_run = _table_run_indexes(lines)
    for index, raw in enumerate(lines):
        normalised = _normalise_layout_line(raw)
        if not normalised:
            active_kind = None
            continue
        if _looks_like_schema(raw):
            kind = "schema"
            active_kind = kind
        elif _looks_like_code(raw):
            kind = "code"
            active_kind = kind
        elif active_kind == "schema" and _looks_like_indented_schema(raw):
            kind = active_kind
        elif active_kind and _looks_like_indented_code(raw):
            kind = active_kind
        else:
            kind = None
            active_kind = None
        if kind:
            classifications[normalised] = kind
        elif index in table_run:
            classifications[normalised] = "table"
    return classifications


def _normalise_layout_line(line: str) -> str:
    return " ".join(line.split())


_SPLIT_WORD = re.compile(r"\b[A-Za-z]{1,16}\s+[a-z]\b")


def _preferred_reading_text(plain: str, layout: str) -> str:
    """Prefer layout extraction only when it improves broken proportional text."""
    if not layout.strip():
        return plain
    plain_compact = re.sub(r"\s+", "", plain)
    layout_compact = re.sub(r"\s+", "", layout)
    # Layout mode can omit floating text in some valid PDFs. It must retain the
    # great majority of the ordinary extract before it is permitted to replace
    # it as the narration/translation source.
    if plain_compact and len(layout_compact) < len(plain_compact) * 0.8:
        return plain
    if len(_SPLIT_WORD.findall(layout)) >= len(_SPLIT_WORD.findall(plain)):
        return plain
    # PDF layout extraction frequently applies the same horizontal margin to
    # every prose line. The cleaner's indentation heuristic would mistake that
    # margin for a code block, so remove only outer line padding. Layout-derived
    # code/schema classifications still preserve genuine technical blocks.
    return "\n".join(_normalise_layout_line(line) for line in layout.splitlines())


def _looks_like_code(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.search(
            r"(?:^\s*(?:def\s+\w+|class\s+\w+|function\s+\w+\s*\(|func\s+\w+|import\s+\w+|SELECT\b|INSERT\b|UPDATE\b|DELETE\b|CREATE\s+TABLE\b)|\b(?:char|short|int|long|float|double|void|size_t|u?int\d*_t|struct\s+\w+|enum\s+\w+)\s*\*?\s*\w+(?:\s*\[[^\]]+\])?\s*;|\b(?:struct|enum)[A-Za-z_]\w*\s*\*?\w+|^\s*\d+\s+.*[=;{}]|[{}]|=>|(?:^|\s)//|\w+\s*\([^)]*\)\s*[;{]|\breturn\s*\w*\s*;|(?:^|\s)(?:\$|prompt>)\s*)",
            stripped,
            re.IGNORECASE,
        )
    )


def _looks_like_schema(line: str) -> bool:
    return bool(
        re.search(
            r"(?:\b(?:CREATE\s+TABLE|ALTER\s+TABLE|CREATE\s+(?:UNIQUE\s+)?INDEX)\b|^\s*(?:PRIMARY|FOREIGN)\s+KEY\s*\()",
            line,
            re.IGNORECASE,
        )
    )


def _looks_like_indented_code(line: str) -> bool:
    leading = len(line) - len(line.lstrip())
    return leading >= 4 and bool(re.search(r"[=\[\]{};]|\breturn\b", line))


def _looks_like_indented_schema(line: str) -> bool:
    return len(line) - len(line.lstrip()) >= 4 and bool(line.strip())


def _table_run_indexes(lines: list[str]) -> set[int]:
    """Find pipe-delimited or genuinely aligned rows, requiring a multi-row run."""
    candidate: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        # Contents pages use dotted leaders to align a title and its page
        # number. They are not data tables and should remain readable prose.
        if re.search(r"(?:\.\s*){5,}", stripped):
            continue
        if "|" in stripped or "\t" in line:
            candidate.append((index, -1))
            continue
        starts = [match.start() for match in re.finditer(r"\S+", line)]
        # A wide, repeated second-column origin distinguishes a table from
        # ordinary justified prose, whose next word follows its first closely.
        if len(starts) >= 2 and starts[1] - starts[0] >= 16:
            candidate.append((index, starts[1]))
    runs: set[int] = set()
    for (left_index, left_column), (right_index, right_column) in zip(candidate, candidate[1:]):
        # Digit widths can move an otherwise aligned column by a few
        # characters. A six-character allowance catches table rows such as
        # `0`, `100`, and `1000` while remaining far narrower than prose.
        same_column = left_column == -1 or right_column == -1 or abs(left_column - right_column) <= 6
        if right_index == left_index + 1 and same_column:
            runs.update((left_index, right_index))
    # Once a genuine multi-row run is found, retain adjoining candidate header
    # and data rows even when their column origins differ slightly. This lets
    # the spoken table summary name headings rather than start at row three.
    candidate_indexes = {index for index, _ in candidate}
    expanded = set(runs)
    for index in sorted(runs):
        previous = index - 1
        while previous in candidate_indexes:
            expanded.add(previous)
            previous -= 1
        following = index + 1
        while following in candidate_indexes:
            expanded.add(following)
            following += 1
    return expanded
