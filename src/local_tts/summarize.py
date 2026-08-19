"""Bounded, reference-aware local study summaries for selected PDF pages."""

from __future__ import annotations

import gc
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import CleanupOptions, ResourceSettings
from .pdf import Chapter, ExtractedPage, PDFDocument
from .text import clean_page

DEFAULT_SUMMARY_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
_CHAPTER_REFERENCE = re.compile(r"\b(?:chapter|chap\.)\s*(\d{1,3})\b", re.IGNORECASE)


@dataclass(frozen=True)
class SummaryOptions:
    """Bounded local-LM controls; character caps protect unified memory."""

    model: str = DEFAULT_SUMMARY_MODEL
    max_context_chars: int = 32_000
    max_output_tokens: int = 480
    max_reference_chapters: int = 3


@dataclass(frozen=True)
class SummarySource:
    role: str
    title: str
    pages: tuple[int, ...]
    excerpt: str


@dataclass(frozen=True)
class SummaryContext:
    selected_pages: tuple[int, ...]
    current_chapters: tuple[Chapter, ...]
    previous_chapter: Chapter | None
    referenced_chapters: tuple[Chapter, ...]
    sources: tuple[SummarySource, ...]
    prompt: str

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_pages": list(self.selected_pages),
            "current_chapters": [_chapter_dict(chapter) for chapter in self.current_chapters],
            "previous_chapter": _chapter_dict(self.previous_chapter) if self.previous_chapter else None,
            "referenced_chapters": [_chapter_dict(chapter) for chapter in self.referenced_chapters],
            "sources": [
                {
                    "role": source.role,
                    "title": source.title,
                    "pages": list(source.pages),
                    "excerpt_chars": len(source.excerpt),
                }
                for source in self.sources
            ],
        }


@dataclass(frozen=True)
class SummaryResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    mlx_active_mb: float
    mlx_peak_mb: float

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["wall_seconds"] = round(self.wall_seconds, 2)
        values["mlx_active_mb"] = round(self.mlx_active_mb, 1)
        values["mlx_peak_mb"] = round(self.mlx_peak_mb, 1)
        return values


class MLXSummaryBackend:
    """A lazy, local-only-after-setup MLX-LM summarization adapter."""

    name = "mlx-lm"

    def __init__(self, options: SummaryOptions) -> None:
        self.options = options
        self._model = None
        self._tokenizer = None
        self._memory_limit_bytes: int | None = None

    def configure(self, resources: ResourceSettings) -> None:
        try:
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - installation concern
            raise RuntimeError("MLX is unavailable; install the normal project dependencies") from exc
        self._memory_limit_bytes = resources.memory_gb * 1024**3
        mx.set_memory_limit(self._memory_limit_bytes)
        mx.set_cache_limit(self._memory_limit_bytes)

    def summarize(self, context: SummaryContext) -> SummaryResult:
        model, tokenizer = self._load()
        import mlx.core as mx

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful technical study companion. Use only the supplied PDF excerpts. "
                    "Do not invent facts, citations, or chapter links."
                ),
            },
            {"role": "user", "content": context.prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_tokens = len(tokenizer.encode(prompt))
        mx.synchronize()
        mx.reset_peak_memory()
        started = time.perf_counter()
        try:
            from mlx_lm import generate
            from mlx_lm.generate import make_sampler
        except ImportError as exc:  # pragma: no cover - _load catches first
            raise RuntimeError("summary support is unavailable; run `uv sync --extra summarize`") from exc
        text = str(
            generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.options.max_output_tokens,
                sampler=make_sampler(temp=0.0),
                verbose=False,
            )
        ).strip()
        mx.synchronize()
        if not text:
            raise RuntimeError("the local summary model produced no text")
        return SummaryResult(
            text=text,
            model=self.options.model,
            input_tokens=input_tokens,
            output_tokens=len(tokenizer.encode(text)),
            wall_seconds=time.perf_counter() - started,
            mlx_active_mb=mx.get_active_memory() / 1024**2,
            mlx_peak_mb=mx.get_peak_memory() / 1024**2,
        )

    def _load(self):
        if self._model is None or self._tokenizer is None:
            try:
                from mlx_lm import load
            except ImportError as exc:
                raise RuntimeError("summary support is unavailable; run `uv sync --extra summarize`") from exc
            self._model, self._tokenizer = load(self._resolve_model_path())
        return self._model, self._tokenizer

    def _resolve_model_path(self) -> str:
        model = self.options.model
        if Path(model).is_dir():
            return model
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return snapshot_download(repo_id=model, local_files_only=True)
        except LocalEntryNotFoundError:
            # First setup deliberately downloads the selected local model once.
            return snapshot_download(repo_id=model)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.synchronize()
            mx.clear_cache()
        except ImportError:  # pragma: no cover - only when installation failed
            pass


def build_summary_context(
    document: PDFDocument,
    selected: list[ExtractedPage],
    chapters: list[Chapter],
    cleanup: CleanupOptions,
    options: SummaryOptions,
) -> SummaryContext:
    """Gather selected material plus bounded chapter/reference context."""
    _validate_options(options)
    selected_pages = tuple(page.number for page in selected)
    if not selected_pages:
        raise ValueError("cannot summarize an empty page selection")
    current = tuple(chapter for chapter in chapters if _overlaps(chapter, selected_pages))
    previous = _previous_chapter(chapters, current, min(selected_pages))
    referenced = _referenced_chapters(selected, chapters, current, previous, options.max_reference_chapters)

    selected_limit = max(2_000, int(options.max_context_chars * 0.62))
    previous_limit = max(1_000, int(options.max_context_chars * 0.18))
    remaining_reference_budget = max(0, options.max_context_chars - selected_limit - previous_limit)
    reference_limit = remaining_reference_budget // max(1, len(referenced)) if referenced else 0

    sources = [
        SummarySource(
            "selected pages",
            _current_title(current, selected_pages),
            selected_pages,
            _excerpt(_clean_pages(selected, cleanup), selected_limit),
        )
    ]
    if previous:
        pages = list(range(previous.start_page, previous.end_page + 1))
        sources.append(
            SummarySource(
                "previous chapter",
                previous.title,
                tuple(pages),
                _excerpt(_clean_pages(document.extract_pages(pages), cleanup), previous_limit),
            )
        )
    for chapter in referenced:
        pages = list(range(chapter.start_page, chapter.end_page + 1))
        sources.append(
            SummarySource(
                "referenced chapter",
                chapter.title,
                tuple(pages),
                _excerpt(_clean_pages(document.extract_pages(pages), cleanup), reference_limit),
            )
        )
    prompt = _prompt(tuple(sources), selected_pages)
    return SummaryContext(selected_pages, current, previous, referenced, tuple(sources), prompt)


def write_summary(output_dir: Path, label: str, context: SummaryContext, result: SummaryResult) -> tuple[Path, Path]:
    """Persist inspectable prose and machine-readable provenance beside audio."""
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{label}-summary.md"
    json_path = output_dir / f"{label}-summary.json"
    scope = ", ".join(str(page) for page in context.selected_pages)
    markdown = (
        f"# Study conclusion: {label}\n\n"
        f"Selected PDF pages: {scope}\n\n"
        f"Model: `{result.model}`\n\n"
        "## Conclusion\n\n"
        f"{result.text}\n"
    )
    _atomic_write(markdown_path, markdown)
    _atomic_write(
        json_path,
        json.dumps({"context": context.as_dict(), "result": result.as_dict()}, indent=2, sort_keys=True) + "\n",
    )
    return markdown_path, json_path


def _clean_pages(pages: list[ExtractedPage], cleanup: CleanupOptions) -> str:
    return "\n\n".join(
        clean_page(page.text, cleanup, layout_kinds=page.technical_lines) for page in pages
    )


def _excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    first = int(limit * 0.7)
    last = max(1, limit - first)
    return f"{text[:first].rstrip()}\n[... excerpt shortened locally ...]\n{text[-last:].lstrip()}"


def _overlaps(chapter: Chapter, pages: tuple[int, ...]) -> bool:
    return chapter.start_page <= max(pages) and chapter.end_page >= min(pages)


def _previous_chapter(chapters: list[Chapter], current: tuple[Chapter, ...], first_page: int) -> Chapter | None:
    current_numbers = {chapter.number for chapter in current}
    eligible = [chapter for chapter in chapters if chapter.number not in current_numbers and chapter.end_page < first_page]
    return eligible[-1] if eligible else None


def _referenced_chapters(
    selected: list[ExtractedPage],
    chapters: list[Chapter],
    current: tuple[Chapter, ...],
    previous: Chapter | None,
    maximum: int,
) -> tuple[Chapter, ...]:
    by_number = {chapter.number: chapter for chapter in chapters}
    excluded = {chapter.number for chapter in current}
    if previous:
        excluded.add(previous.number)
    numbers = {
        int(match.group(1))
        for page in selected
        for match in _CHAPTER_REFERENCE.finditer(page.text)
        if int(match.group(1)) in by_number and int(match.group(1)) not in excluded
    }
    return tuple(by_number[number] for number in sorted(numbers)[:maximum])


def _current_title(current: tuple[Chapter, ...], pages: tuple[int, ...]) -> str:
    if current:
        return "; ".join(f"Chapter {chapter.number}: {chapter.title}" for chapter in current)
    return f"Selected pages {min(pages)}-{max(pages)}"


def _prompt(sources: tuple[SummarySource, ...], selected_pages: tuple[int, ...]) -> str:
    sections = "\n\n".join(
        f"[{source.role.upper()}: {source.title}; PDF pages {min(source.pages)}-{max(source.pages)}]\n{source.excerpt}"
        for source in sources
        if source.excerpt
    )
    return (
        f"Write a conclusive, listener-friendly study summary of the selected PDF pages {min(selected_pages)}-{max(selected_pages)}. "
        "The selected-pages excerpt is the evidence to summarize. Previous and referenced chapter excerpts are context only: "
        "use them solely to explain a connection when it is explicit. Explain the central idea, causal mechanism, key terms, "
        "and practical takeaway. Keep it to 4-7 short paragraphs, use plain prose suitable for text-to-speech, and do not repeat yourself.\n\n"
        f"{sections}"
    )


def _chapter_dict(chapter: Chapter) -> dict[str, object]:
    return {"number": chapter.number, "title": chapter.title, "start_page": chapter.start_page, "end_page": chapter.end_page}


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _validate_options(options: SummaryOptions) -> None:
    if options.max_context_chars < 4_000:
        raise ValueError("summary context must be at least 4000 characters")
    if not 64 <= options.max_output_tokens <= 4_096:
        raise ValueError("summary max output tokens must be between 64 and 4096")
    if not 0 <= options.max_reference_chapters <= 8:
        raise ValueError("summary references must be between 0 and 8")
