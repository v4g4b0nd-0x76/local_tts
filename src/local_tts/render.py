"""Bounded PDF-to-audio rendering with crash-safe append-only checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .backends.base import TTSBackend
from .models import CleanupOptions, RenderOptions, ResourceSettings
from .pdf import ExtractedPage
from .text import apply_pronunciations, chunk_text, clean_page, repeated_margin_lines

_STATE_VERSION = 2


@dataclass(frozen=True)
class RenderReport:
    output: Path
    chunks_total: int
    chunks_synthesized: int
    audio_seconds: float
    wall_seconds: float
    detected_code_lines: int
    detected_schema_lines: int
    detected_table_lines: int


@dataclass(frozen=True)
class _Checkpoint:
    page: int
    text_hash: str
    duration_seconds: float
    end_bytes: int
    empty: bool = False


class _CheckpointStore:
    """A durable PCM stream plus an append-only JSONL chunk journal.

    The write order is intentional: durable PCM bytes first, then a durable
    checkpoint. After a crash, resume truncates any unjournaled tail before
    appending. This avoids thousands of WAV files and rewriting an O(n) JSON
    document for every chunk while preserving chunk-level restart safety.
    """

    def __init__(self, job_dir: Path, resume: bool) -> None:
        self.job_dir = job_dir
        self.pcm_path = job_dir / "audio.s16le"
        self.journal_path = job_dir / "checkpoints.jsonl"
        self.manifest_path = job_dir / "manifest.json"
        self.records: dict[int, _Checkpoint] = {}
        self.completed = False
        job_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self._read_journal()
            self._repair_pcm_tail()
        else:
            self.pcm_path.write_bytes(b"")
            self.journal_path.write_text("")

    def write_manifest(
        self,
        resources: ResourceSettings,
        cleanup: CleanupOptions,
        options: RenderOptions,
        book_metadata: dict[str, object] | None,
        layout_summary: dict[str, int],
    ) -> None:
        manifest = {
            "version": _STATE_VERSION,
            "audio_encoding": "s16le",
            "sample_rate": options.sample_rate,
            "resources": resources.as_dict(),
            "cleanup": asdict(cleanup),
            "render_options": asdict(options),
            "book_metadata": book_metadata or {},
            "layout_summary": layout_summary,
        }
        _atomic_json(self.manifest_path, manifest)

    def append_audio(self, audio: np.ndarray) -> int:
        pcm = _as_pcm16(audio)
        start = self.pcm_path.stat().st_size if self.pcm_path.exists() else 0
        with self.pcm_path.open("ab") as handle:
            handle.write(pcm)
            handle.flush()
            os.fsync(handle.fileno())
        return start + len(pcm)

    def append_checkpoint(self, index: int, checkpoint: _Checkpoint) -> None:
        payload = {
            "event": "chunk",
            "index": index,
            "page": checkpoint.page,
            "text_hash": checkpoint.text_hash,
            "duration_seconds": checkpoint.duration_seconds,
            "end_bytes": checkpoint.end_bytes,
            "empty": checkpoint.empty,
        }
        self._append_event(payload)
        self.records[index] = checkpoint

    def mark_complete(self, output: Path) -> None:
        self._append_event({"event": "complete", "output": str(output)})
        self.completed = True

    def _append_event(self, payload: dict[str, object]) -> None:
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _read_journal(self) -> None:
        if not self.journal_path.exists():
            return
        lines = self.journal_path.read_text().splitlines()
        for line_number, line in enumerate(lines, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave the final buffered line incomplete. It has
                # no durable checkpoint, so safely ignore it.
                if line_number == len(lines):
                    continue
                raise ValueError("checkpoint journal is corrupt before its final line")
            if event.get("event") == "chunk":
                index = int(event["index"])
                self.records[index] = _Checkpoint(
                    page=int(event["page"]),
                    text_hash=str(event["text_hash"]),
                    duration_seconds=float(event["duration_seconds"]),
                    end_bytes=int(event["end_bytes"]),
                    empty=bool(event.get("empty", False)),
                )
            elif event.get("event") == "complete":
                self.completed = True
        expected = set(range(1, len(self.records) + 1))
        if set(self.records) != expected:
            raise ValueError("checkpoint journal has non-contiguous chunks; rerun without --resume")

    def _repair_pcm_tail(self) -> None:
        if not self.records:
            return
        if not self.pcm_path.exists():
            raise ValueError("checkpoint journal exists but its PCM stream is missing")
        expected_end = max(record.end_bytes for record in self.records.values())
        current_size = self.pcm_path.stat().st_size
        if current_size < expected_end:
            raise ValueError("PCM stream is shorter than the last durable checkpoint")
        if current_size > expected_end:
            with self.pcm_path.open("r+b") as handle:
                handle.truncate(expected_end)
                handle.flush()
                os.fsync(handle.fileno())


def render(
    extracted_pages: list[tuple[int, str] | ExtractedPage],
    output_dir: Path,
    label: str,
    backend: TTSBackend,
    resources: ResourceSettings,
    cleanup: CleanupOptions,
    options: RenderOptions,
    book_metadata: dict[str, object] | None = None,
) -> RenderReport:
    """Render selected pages with bounded CPU preparation and durable resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir = output_dir / f".{label}.local-tts"
    store = _CheckpointStore(job_dir, options.resume)
    layout_summary = _layout_summary(extracted_pages)
    if not options.resume:
        store.write_manifest(resources, cleanup, options, book_metadata, layout_summary)
    margins = repeated_margin_lines([_page_parts(page)[1] for page in extracted_pages]) if cleanup.headers_footers else set()
    started = time.perf_counter()
    synthesized = 0
    total_audio = 0.0
    chunks_total = 0

    # The producer is bounded by `prefetch`: while MLX synthesizes the current
    # chunk, cleanup workers prepare the next pages without retaining a book's
    # worth of cleaned text in memory.
    for page, cleaned in _cleaned_pages(extracted_pages, cleanup, margins, resources):
        for text in chunk_text(cleaned, resources.chunk_chars):
            chunks_total += 1
            spoken_text = apply_pronunciations(text, options.pronunciations)
            text_hash = _text_hash(page, spoken_text, options)
            checkpoint = store.records.get(chunks_total)
            if options.resume and checkpoint is not None:
                _verify_checkpoint(chunks_total, checkpoint, page, text_hash)
                total_audio += checkpoint.duration_seconds
                continue

            result = backend.synthesize(spoken_text, voice=options.voice, speed=options.speed, sample_rate=options.sample_rate)
            audio = _with_pause(result.audio, result.sample_rate, options.chunk_pause_ms)
            duration = len(audio) / result.sample_rate
            end_bytes = store.append_audio(audio) if audio.size else _pcm_size(store.pcm_path)
            store.append_checkpoint(
                chunks_total,
                _Checkpoint(page, text_hash, duration, end_bytes, empty=result.audio.size == 0),
            )
            total_audio += duration
            synthesized += 1

    if not chunks_total:
        raise ValueError("no readable text remained after extraction and cleanup")
    if options.resume and len(store.records) > chunks_total:
        raise ValueError("resume checkpoint has more chunks than the selected text; rerun without --resume")

    output = output_dir / f"{label}.{options.audio_format}"
    if not (options.resume and store.completed and output.exists() and synthesized == 0):
        _encode(
            store.pcm_path,
            output,
            options.audio_format,
            options.sample_rate,
            resources.cpu_threads,
            book_metadata,
            label,
        )
        store.mark_complete(output)
    elapsed = time.perf_counter() - started
    return RenderReport(
        output, chunks_total, synthesized, total_audio, elapsed,
        layout_summary["code_lines"], layout_summary["schema_lines"], layout_summary["table_lines"],
    )


def _cleaned_pages(
    extracted_pages: list[tuple[int, str] | ExtractedPage],
    cleanup: CleanupOptions,
    margins: set[str],
    resources: ResourceSettings,
) -> Iterator[tuple[int, str]]:
    """Yield ordered cleaned pages while retaining only a bounded work queue."""
    source = iter(extracted_pages)
    pending: deque[tuple[int, Future[str]]] = deque()
    with ThreadPoolExecutor(max_workers=resources.cpu_threads, thread_name_prefix="local-tts-clean") as executor:
        def fill() -> None:
            while len(pending) < resources.prefetch:
                try:
                    item = next(source)
                except StopIteration:
                    return
                page, text, layout_kinds = _page_parts(item)
                pending.append((page, executor.submit(clean_page, text, cleanup, margins, layout_kinds)))

        fill()
        while pending:
            page, future = pending.popleft()
            fill()
            cleaned = future.result()
            if cleaned:
                yield page, cleaned


def _text_hash(page: int, text: str, options: RenderOptions) -> str:
    pronunciations = "\x1e".join(f"{source}\x1f{spoken}" for source, spoken in options.pronunciations)
    data = f"{page}\0{options.voice}\0{options.speed}\0{options.sample_rate}\0{options.chunk_pause_ms}\0{pronunciations}\0{text}".encode()
    return hashlib.sha256(data).hexdigest()


def _verify_checkpoint(index: int, checkpoint: _Checkpoint, page: int, text_hash: str) -> None:
    if checkpoint.page != page or checkpoint.text_hash != text_hash:
        raise ValueError(
            f"resume checkpoint {index} does not match the selected text or render settings; rerun without --resume"
        )


def _as_pcm16(audio: np.ndarray) -> bytes:
    samples = np.asarray(audio, dtype=np.float32)
    return np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2", copy=False).tobytes()


def _with_pause(audio: np.ndarray, sample_rate: int, pause_ms: int) -> np.ndarray:
    if pause_ms <= 0 or audio.size == 0:
        return audio
    pause = np.zeros(round(sample_rate * pause_ms / 1000), dtype=np.float32)
    return np.concatenate((np.asarray(audio, dtype=np.float32), pause))


def _page_parts(item: tuple[int, str] | ExtractedPage) -> tuple[int, str, dict[str, str]]:
    if isinstance(item, ExtractedPage):
        return item.number, item.text, item.technical_lines
    page, text = item
    return page, text, {}


def _layout_summary(items: list[tuple[int, str] | ExtractedPage]) -> dict[str, int]:
    pages = [item for item in items if isinstance(item, ExtractedPage)]
    return {
        "layout_aware_pages": len(pages),
        "code_lines": sum(page.layout_stats.code_lines for page in pages),
        "schema_lines": sum(page.layout_stats.schema_lines for page in pages),
        "table_lines": sum(page.layout_stats.table_lines for page in pages),
    }


def _pcm_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _encode(
    pcm_path: Path,
    output: Path,
    audio_format: str,
    sample_rate: int,
    cpu_threads: int,
    book_metadata: dict[str, object] | None,
    label: str,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for final audio encoding")
    if _pcm_size(pcm_path) == 0:
        raise ValueError("no audio chunks were synthesized")
    codecs = {
        "m4a": ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"],
        "m4b": ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"],
        "mp3": ["-c:a", "libmp3lame", "-b:a", "96k"],
    }
    if audio_format not in codecs:
        raise ValueError("audio format must be one of: m4a, m4b, mp3")
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", str(pcm_path),
        "-threads", str(cpu_threads), *_ffmpeg_metadata(book_metadata, label), *codecs[audio_format], str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _ffmpeg_metadata(book_metadata: dict[str, object] | None, label: str) -> list[str]:
    if not book_metadata:
        return []
    title = str(book_metadata.get("title") or "")
    values = {
        "title": f"{title} - {label}" if title else label,
        "album": title,
        "artist": str(book_metadata.get("author") or ""),
        "comment": str(book_metadata.get("subject") or ""),
    }
    return [argument for key, value in values.items() if value for argument in ("-metadata", f"{key}={value}")]
