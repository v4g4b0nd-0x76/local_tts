"""Durable, reviewable local English-to-Persian PDF translation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
from typing import Callable, Protocol

from .models import CleanupOptions, ResourceSettings
from .pdf import ExtractedPage
from .progress import ProgressEvent
from .summarize import SummaryResult
from .text import chunk_text, clean_page, repeated_margin_lines

DEFAULT_TRANSLATION_BACKEND = "nllb"
DEFAULT_NLLB_MODEL = "facebook/nllb-200-distilled-600M"


@dataclass(frozen=True)
class TranslationOptions:
    model: str
    backend: str = DEFAULT_TRANSLATION_BACKEND
    max_source_chars: int = 1_200
    max_output_tokens: int = 1_024
    protected_terms: tuple[str, ...] = ()
    glossary: tuple[tuple[str, str], ...] = ()
    transliterations: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TranslationSegment:
    index: int
    page: int
    source: str
    translation: str
    generation: SummaryResult


@dataclass(frozen=True)
class TranslationReport:
    language: str
    backend: str
    model: str
    segments: tuple[TranslationSegment, ...]
    sidecar: Path


class PersianTranslator(Protocol):
    def translate_to_persian(
        self,
        source_text: str,
        *,
        max_tokens: int,
        glossary: tuple[tuple[str, str], ...] = (),
    ) -> SummaryResult: ...


def translate_pages_to_persian(
    pages: list[ExtractedPage],
    output_dir: Path,
    label: str,
    translator: PersianTranslator,
    resources: ResourceSettings,
    cleanup: CleanupOptions,
    options: TranslationOptions,
    *,
    resume: bool,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> TranslationReport:
    """Translate cleaned PDF text with one fsync'd translation record per chunk.

    Translation can be much slower than speech. The journal ensures `--resume`
    reuses the exact prior Persian text instead of asking an LLM to translate a
    completed segment again with slightly different wording.
    """
    _validate_options(options)
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = _TranslationJournal(output_dir / f".{label}.local-tts" / "translation.jsonl", resume, options)
    margins = repeated_margin_lines([page.text for page in pages]) if cleanup.headers_footers else set()
    segments: list[TranslationSegment] = []
    index = 0
    started = time.perf_counter()
    effective_glossary = _effective_glossary(options)
    allowed_transliterations = _allowed_transliterations(options)
    for page_index, page in enumerate(pages, start=1):
        cleaned = clean_page(page.text, cleanup, margins=margins, layout_kinds=page.technical_lines)
        sources = list(chunk_text(cleaned, min(resources.chunk_chars, options.max_source_chars)))
        for source_index, source in enumerate(sources, start=1):
            index += 1
            source_hash = _source_hash(page.number, source, options)
            existing = journal.records.get(index)
            if existing is not None:
                if existing["page"] != page.number or existing["source_hash"] != source_hash:
                    raise ValueError("translation checkpoint does not match the selected PDF text; rerun without --resume")
                segments.append(_segment_from_record(existing))
                _emit_progress(
                    progress,
                    "resuming translation",
                    page_index - 1 + source_index / len(sources),
                    len(pages),
                    index,
                    0.0,
                    started,
                    f"page {page.number}, segment {index}",
                )
                continue
            generated = translator.translate_to_persian(
                source,
                max_tokens=options.max_output_tokens,
                glossary=effective_glossary,
            )
            translated = apply_transliterations(normalize_persian(generated.text), allowed_transliterations)
            if not _contains_arabic_script(translated):
                raise RuntimeError("the local translator did not return Persian text; try a larger translation model")
            # Keep the sidecar and resumed record byte-for-byte consistent with
            # the normalized text we actually give to the Persian TTS engine.
            segment = TranslationSegment(index, page.number, source, translated, replace(generated, text=translated))
            journal.append(segment, source_hash)
            segments.append(segment)
            _emit_progress(
                progress,
                "translating",
                page_index - 1 + source_index / len(sources),
                len(pages),
                index,
                0.0,
                started,
                f"page {page.number}, segment {index}",
            )
    if not segments:
        raise ValueError("no readable text remained after cleanup for translation")
    if resume and len(journal.records) > len(segments):
        raise ValueError("translation checkpoint has more chunks than the selected text; rerun without --resume")
    sidecar = output_dir / f"{label}.translation.json"
    _atomic_write(
        sidecar,
        json.dumps(
            {
                "language": "fa",
                "backend": options.backend,
                "model": options.model,
                "options": asdict(options),
                "segments": [_segment_dict(segment) for segment in segments],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
    )
    _emit_progress(progress, "writing translation", len(pages), len(pages), index, 0.0, started, "sidecar saved")
    return TranslationReport("fa", options.backend, options.model, tuple(segments), sidecar)


def normalize_persian(text: str) -> str:
    """Normalize Arabic presentation variants into Persian spelling for TTS."""
    normalized = text.translate(str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "هٔ", "ة": "ه"}))
    normalized = normalized.replace("ـ", "")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    return normalized.strip()


def apply_transliterations(text: str, replacements: tuple[tuple[str, str], ...]) -> str:
    """Replace known Latin proper names with user-approved Persian spellings."""
    result = text
    for source, target in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if not source.strip() or not target.strip():
            continue
        result = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, result, flags=re.IGNORECASE)
    return result


def _contains_arabic_script(text: str) -> bool:
    return any("\u0600" <= character <= "\u06ff" for character in text)


def _validate_options(options: TranslationOptions) -> None:
    if options.backend not in {"nllb", "qwen"}:
        raise ValueError("translation backend must be nllb or qwen")
    if not options.model.strip():
        raise ValueError("translation model must be non-empty")
    if not 200 <= options.max_source_chars <= 3_000:
        raise ValueError("translation source chunk size must be between 200 and 3000 characters")
    if not 64 <= options.max_output_tokens <= 2_048:
        raise ValueError("translation max output tokens must be between 64 and 2048")
    if any(not term.strip() for term in options.protected_terms):
        raise ValueError("translation protected terms must be non-empty")
    if any(not source.strip() or not target.strip() for source, target in options.transliterations):
        raise ValueError("translation transliterations must map non-empty text to non-empty Persian spellings")


def _effective_glossary(options: TranslationOptions) -> tuple[tuple[str, str], ...]:
    """Give protected source wording precedence over a Persian glossary entry."""
    protected: list[str] = []
    protected_keys: set[str] = set()
    for term in options.protected_terms:
        key = term.casefold()
        if key not in protected_keys:
            protected.append(term)
            protected_keys.add(key)
    return tuple((term, term) for term in protected) + tuple(
        (source, target) for source, target in options.glossary if source.casefold() not in protected_keys
    )


def _allowed_transliterations(options: TranslationOptions) -> tuple[tuple[str, str], ...]:
    """Do not replace an exact source term that the reader chose to preserve."""
    protected_keys = {term.casefold() for term in options.protected_terms}
    return tuple((source, target) for source, target in options.transliterations if source.casefold() not in protected_keys)


def _emit_progress(
    callback: Callable[[ProgressEvent], None] | None,
    phase: str,
    completed: float,
    total: float,
    units_completed: int,
    audio_seconds: float,
    started: float,
    detail: str,
) -> None:
    if callback is not None:
        callback(
            ProgressEvent(
                phase,
                completed,
                total,
                units_completed,
                audio_seconds,
                time.perf_counter() - started,
                detail,
            )
        )


def _source_hash(page: int, source: str, options: TranslationOptions) -> str:
    payload = json.dumps(
        {"page": page, "source": source, "options": asdict(options)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _segment_dict(segment: TranslationSegment) -> dict[str, object]:
    return {
        "index": segment.index,
        "page": segment.page,
        "source": segment.source,
        "translation": segment.translation,
        "generation": segment.generation.as_dict(),
    }


def _segment_from_record(record: dict[str, object]) -> TranslationSegment:
    generation = record["generation"]
    if not isinstance(generation, dict):
        raise ValueError("translation checkpoint generation data is invalid")
    return TranslationSegment(
        int(record["index"]),
        int(record["page"]),
        str(record["source"]),
        str(record["translation"]),
        SummaryResult(
            text=str(record["translation"]),
            model=str(generation["model"]),
            input_tokens=int(generation["input_tokens"]),
            output_tokens=int(generation["output_tokens"]),
            wall_seconds=float(generation["wall_seconds"]),
            mlx_active_mb=float(generation["mlx_active_mb"]),
            mlx_peak_mb=float(generation["mlx_peak_mb"]),
        ),
    )


class _TranslationJournal:
    def __init__(self, path: Path, resume: bool, options: TranslationOptions) -> None:
        self.path = path
        self.records: dict[int, dict[str, object]] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            self._read(options)
        else:
            path.write_text("")

    def _read(self, options: TranslationOptions) -> None:
        if not self.path.exists():
            return
        lines = self.path.read_text().splitlines()
        for line_number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(lines):
                    continue
                raise ValueError("translation journal is corrupt before its final line")
            if record.get("event") != "translation":
                raise ValueError("translation journal contains an unknown record")
            index = int(record["index"])
            self.records[index] = record
        expected = set(range(1, len(self.records) + 1))
        if set(self.records) != expected:
            raise ValueError("translation journal has non-contiguous chunks; rerun without --resume")

    def append(self, segment: TranslationSegment, source_hash: str) -> None:
        record = {
            "event": "translation",
            **_segment_dict(segment),
            "source_hash": source_hash,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[segment.index] = record


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)
