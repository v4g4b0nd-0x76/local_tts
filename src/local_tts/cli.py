"""Command-line interface for PDF selection, rendering, and benchmarking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .backends import KokoroMLXBackend, NLLBPersianTranslator, PiperFarsiBackend
from .backends.piper_farsi import DEFAULT_FARSI_VOICE, default_farsi_model_dir, piper_voice_path
from .benchmark import benchmark_candidates
from .config import config_value, load_config
from .models import CleanupOptions, PROFILES, ProfileName, RenderOptions, ResourceSettings
from .pdf import PDFDocument, chapter_pages, page_range
from .progress import ProgressEvent, TerminalProgress
from .render import ScriptLine, render, render_script
from .summarize import (
    DEFAULT_SUMMARY_MODEL,
    MLXSummaryBackend,
    PodcastOptions,
    SummaryOptions,
    build_summary_context,
    write_podcast,
    write_summary,
)
from .translation import DEFAULT_NLLB_MODEL, DEFAULT_TRANSLATION_BACKEND, TranslationOptions, translate_pages_to_persian


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-tts", description="Fully local PDF-to-speech study tool")
    subparsers = parser.add_subparsers(dest="command")
    bench = subparsers.add_parser("benchmark", help="measure local TTS throughput")
    _add_resources(bench)
    bench.add_argument("--runs", type=int, default=2, help="measured passes per chunk-size candidate")
    serve = subparsers.add_parser("serve", help="run the loopback streaming TTS API for local readers")
    _add_serve_args(serve)
    setup_viewer = subparsers.add_parser("setup-viewer", help="install the one-time local PDF.js reader assets")
    setup_viewer.add_argument("--force", action="store_true", help="refresh the local PDF.js installation")
    setup_farsi = subparsers.add_parser("setup-farsi", help="download one local Persian Piper voice")
    setup_farsi.add_argument("--voice", default=DEFAULT_FARSI_VOICE, help="Piper voice id, e.g. fa_IR-ganji_adabi-medium")
    setup_farsi.add_argument("--model-dir", type=Path, default=default_farsi_model_dir(), help="local directory for Piper voice files")
    reader = subparsers.add_parser("read", help="open one PDF in the local PDF.js reader with speech controls")
    _add_reader_args(reader)
    book = subparsers.add_parser("book", help="render a PDF")
    _add_book_args(book)
    return parser


def _add_resources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=[item.value for item in ProfileName])
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--batch-size", type=int, help="backend batch size (Kokoro MLX v1 uses 1)")
    parser.add_argument("--prefetch", type=int)
    parser.add_argument("--chunk-chars", type=int)
    parser.add_argument("--memory-gb", type=int)


def _add_book_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--config", type=Path, help="TOML reading-policy and narration configuration")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--chapter", metavar="N")
    select.add_argument("--chapters", metavar="N-M")
    select.add_argument("--pages", metavar="N-M")
    parser.add_argument("--list", action="store_true", help="list page count and extracted outline")
    parser.add_argument("--metadata", action="store_true", help="print embedded PDF/book metadata as JSON")
    parser.add_argument(
        "--summarize", "--summerize", dest="summarize", action="store_true",
        help="write and narrate a local conclusion using selected pages plus bounded chapter context",
    )
    parser.add_argument(
        "--podcast", action="store_true",
        help="write and narrate a local two-voice, simple-language study conversation",
    )
    parser.add_argument("--summary-model", help="local MLX model for --summarize and --podcast")
    parser.add_argument("--summary-context-chars", type=int, help="maximum source characters sent to the local summary model")
    parser.add_argument("--summary-max-tokens", type=int, help="maximum tokens in the concluded summary")
    parser.add_argument("--summary-references", type=int, help="number of explicitly referenced chapters to include (0-8)")
    parser.add_argument("--podcast-host-voice", help="Kokoro host voice (sample American voice: af_bella)")
    parser.add_argument("--podcast-explainer-voice", help="Kokoro explainer voice (sample American voice: am_michael)")
    parser.add_argument("--podcast-max-tokens", type=int, help="maximum tokens in the generated dialogue script")
    parser.add_argument("--podcast-max-turns", type=int, help="maximum alternating host/explainer turns (2-24)")
    parser.add_argument("--podcast-turn-pause-ms", type=int, help="silence after each host/explainer turn")
    parser.add_argument("--translate", choices=["fa"], help="translate the selected PDF text to Persian before narration")
    parser.add_argument("--translation-backend", choices=["nllb", "qwen"], help="local translator: NLLB (literal default) or Qwen (glossary-aware)")
    parser.add_argument("--translation-model", help="local model used by the selected translation backend")
    parser.add_argument("--translation-chunk-chars", type=int, help="maximum English characters in one translation request")
    parser.add_argument("--translation-max-tokens", type=int, help="maximum Persian tokens per translation request")
    parser.add_argument("--farsi-voice", help="installed Piper Persian voice id")
    parser.add_argument("--farsi-model-dir", type=Path, help="local Piper voice directory")
    parser.add_argument("--farsi-noise-scale", type=float, help="Piper variation (lower is steadier; 0-2)")
    parser.add_argument("--farsi-noise-w-scale", type=float, help="Piper duration variation (lower is steadier; 0-2)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--format", choices=["m4a", "m4b", "mp3"])
    parser.add_argument("--voice")
    parser.add_argument("--speed", type=float)
    parser.add_argument("--sample-rate", type=int, choices=[24000, 48000])
    parser.add_argument("--chunk-pause-ms", type=int, help="silence after each synthesized chunk for calmer transitions")
    parser.add_argument("--code", choices=["skip", "read", "explain"])
    parser.add_argument("--schemas", choices=["skip", "read", "explain"])
    parser.add_argument("--tables", choices=["skip", "read", "explain"])
    parser.add_argument("--urls", choices=["skip", "read"])
    parser.add_argument("--citations", choices=["skip", "read"])
    parser.add_argument("--footnotes", choices=["skip", "read"])
    parser.add_argument("--keep-headers-footers", action="store_true", default=None)
    _add_resources(parser)


def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="TOML reader and resource configuration")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (loopback by default)")
    parser.add_argument("--port", type=int, default=8765, help="local HTTP port (default: 8765)")
    parser.add_argument("--allow-network", action="store_true", help="allow a non-loopback bind address")
    parser.add_argument("--allow-origin", action="append", default=[], help="additional browser origin allowed by CORS")
    parser.add_argument("--max-request-chars", type=int, default=50_000, help="maximum text characters per request")
    parser.add_argument("--voice")
    parser.add_argument("--speed", type=float)
    parser.add_argument("--sample-rate", type=int, choices=[24000, 48000])
    parser.add_argument("--chunk-pause-ms", type=int, help="silence after each streamed audio chunk")
    _add_resources(parser)


def _add_reader_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--no-open", action="store_true", help="start the reader without opening a browser tab")
    _add_serve_args(parser)
    parser.add_argument(
        "--theme",
        choices=["kuro-nezumi", "default", "custom"],
        help="reader appearance: Kuro Nezumi, the standard PDF.js appearance, or --theme-css",
    )
    parser.add_argument("--theme-css", type=Path, help="local CSS file used with --theme custom")


def _resources(args: argparse.Namespace) -> ResourceSettings:
    profile = ProfileName(args.profile or ProfileName.BALANCED)
    base = PROFILES.get(profile, PROFILES[ProfileName.BALANCED])
    values = {
        "cpu_threads": args.cpu_threads or base.cpu_threads,
        "batch_size": args.batch_size or base.batch_size,
        "prefetch": args.prefetch or base.prefetch,
        "chunk_chars": args.chunk_chars or base.chunk_chars,
        "memory_gb": args.memory_gb or base.memory_gb,
    }
    if min(values.values()) < 1:
        raise ValueError("resource values must be positive")
    return ResourceSettings(**values)


def _apply_book_config(args: argparse.Namespace) -> None:
    """Apply TOML values only when the corresponding CLI switch was omitted."""
    config = load_config(args.config)

    def assign(attr: str, section: str, key: str, fallback):
        if getattr(args, attr) is None:
            setattr(args, attr, config_value(config, section, key, fallback))

    assign("profile", "resources", "profile", "balanced")
    for name in ("cpu_threads", "batch_size", "prefetch", "chunk_chars", "memory_gb"):
        assign(name, "resources", name, None)
    assign("voice", "reader", "voice", "af_heart")
    assign("speed", "reader", "speed", 1.0)
    assign("sample_rate", "reader", "sample_rate", 24000)
    assign("chunk_pause_ms", "reader", "chunk_pause_ms", 0)
    assign("summary_model", "summary", "model", None)
    assign("summary_context_chars", "summary", "context_chars", 32_000)
    assign("summary_max_tokens", "summary", "max_output_tokens", 480)
    assign("summary_references", "summary", "max_references", 3)
    assign("podcast_host_voice", "podcast", "host_voice", "af_bella")
    assign("podcast_explainer_voice", "podcast", "explainer_voice", "am_michael")
    assign("podcast_max_tokens", "podcast", "max_output_tokens", 640)
    assign("podcast_max_turns", "podcast", "max_turns", 8)
    assign("podcast_turn_pause_ms", "podcast", "turn_pause_ms", 350)
    assign("translate", "translation", "target", None)
    assign("translation_backend", "translation", "backend", DEFAULT_TRANSLATION_BACKEND)
    if args.translation_backend not in {"nllb", "qwen"}:
        raise ValueError("translation backend must be nllb or qwen")
    assign(
        "translation_model",
        "translation",
        "model",
        DEFAULT_NLLB_MODEL if args.translation_backend == "nllb" else DEFAULT_SUMMARY_MODEL,
    )
    assign("translation_chunk_chars", "translation", "chunk_chars", 1_200)
    assign("translation_max_tokens", "translation", "max_output_tokens", 1_024)
    assign("farsi_voice", "translation", "voice", DEFAULT_FARSI_VOICE)
    assign("farsi_model_dir", "translation", "voice_dir", default_farsi_model_dir())
    assign("farsi_noise_scale", "translation", "noise_scale", 0.45)
    assign("farsi_noise_w_scale", "translation", "noise_w_scale", 0.65)
    if isinstance(args.farsi_model_dir, str):
        args.farsi_model_dir = Path(args.farsi_model_dir)
    glossary = config.get("translation", {}).get("glossary", {})
    if not isinstance(glossary, dict) or any(
        not isinstance(source, str) or not isinstance(target, str) for source, target in glossary.items()
    ):
        raise ValueError("[translation.glossary] must map source terms to Persian terms")
    args.translation_glossary = tuple(sorted((source, target) for source, target in glossary.items()))
    protected_terms = config.get("translation", {}).get("protect", {})
    if not isinstance(protected_terms, dict) or any(
        not isinstance(term, str) or enabled is not True for term, enabled in protected_terms.items()
    ):
        raise ValueError("[translation.protect] must map each source term or name to true")
    args.translation_protected_terms = tuple(sorted(protected_terms))
    transliterations = config.get("translation", {}).get("transliteration", {})
    if not isinstance(transliterations, dict) or any(
        not isinstance(source, str) or not isinstance(target, str) for source, target in transliterations.items()
    ):
        raise ValueError("[translation.transliteration] must map Latin names to Persian spellings")
    args.translation_transliterations = tuple(sorted((source, target) for source, target in transliterations.items()))
    assign("output_dir", "output", "directory", Path("output"))
    if isinstance(args.output_dir, str):
        args.output_dir = Path(args.output_dir)
    assign("format", "output", "format", "m4a")
    for name in ("code", "schemas", "tables", "urls", "citations", "footnotes"):
        defaults = {"code": "skip", "schemas": "skip", "tables": "skip", "urls": "skip", "citations": "skip", "footnotes": "skip"}
        assign(name, "cleanup", name, defaults[name])
    if args.keep_headers_footers is None:
        args.keep_headers_footers = not bool(config_value(config, "cleanup", "headers_footers", True))
    if args.speed <= 0:
        raise ValueError("reader speed must be greater than zero")
    if args.sample_rate not in {24000, 48000}:
        raise ValueError("reader sample_rate must be 24000 or 48000")
    if not 0 <= args.chunk_pause_ms <= 2000:
        raise ValueError("reader chunk_pause_ms must be between 0 and 2000")
    if args.summary_context_chars < 4_000:
        raise ValueError("summary context must be at least 4000 characters")
    if not 64 <= args.summary_max_tokens <= 4_096:
        raise ValueError("summary max output tokens must be between 64 and 4096")
    if not 0 <= args.summary_references <= 8:
        raise ValueError("summary references must be between 0 and 8")
    if not args.podcast_host_voice.strip() or not args.podcast_explainer_voice.strip():
        raise ValueError("podcast voices must be non-empty")
    if not 64 <= args.podcast_max_tokens <= 4_096:
        raise ValueError("podcast max output tokens must be between 64 and 4096")
    if not 2 <= args.podcast_max_turns <= 24:
        raise ValueError("podcast max turns must be between 2 and 24")
    if not 0 <= args.podcast_turn_pause_ms <= 4_000:
        raise ValueError("podcast turn_pause_ms must be between 0 and 4000")
    if args.translate is not None and args.translate != "fa":
        raise ValueError("only Persian translation is available in v1")
    if args.translate:
        if args.summarize or args.podcast:
            raise ValueError("--translate cannot yet be combined with --summarize or --podcast")
        if not 200 <= args.translation_chunk_chars <= 3_000:
            raise ValueError("translation chunk size must be between 200 and 3000 characters")
    if not 64 <= args.translation_max_tokens <= 2_048:
        raise ValueError("translation max output tokens must be between 64 and 2048")
    if not 0 <= args.farsi_noise_scale <= 2:
        raise ValueError("Farsi noise scale must be between 0 and 2")
    if not 0 <= args.farsi_noise_w_scale <= 2:
        raise ValueError("Farsi duration noise scale must be between 0 and 2")
    if not args.farsi_voice.strip():
        raise ValueError("Farsi voice must be non-empty")
    args.pronunciations = _pronunciations(config)


def _apply_serve_config(args: argparse.Namespace) -> None:
    """Apply the reader parts of the TOML configuration to the local API."""
    config = load_config(args.config)

    def assign(attr: str, section: str, key: str, fallback):
        if getattr(args, attr) is None:
            setattr(args, attr, config_value(config, section, key, fallback))

    assign("profile", "resources", "profile", "balanced")
    for name in ("cpu_threads", "batch_size", "prefetch", "chunk_chars", "memory_gb"):
        assign(name, "resources", name, None)
    assign("voice", "reader", "voice", "af_heart")
    assign("speed", "reader", "speed", 1.0)
    assign("sample_rate", "reader", "sample_rate", 24_000)
    assign("chunk_pause_ms", "reader", "chunk_pause_ms", 0)
    if args.port not in range(1, 65_536):
        raise ValueError("server port must be between 1 and 65535")
    if args.host not in {"127.0.0.1", "::1", "localhost"} and not args.allow_network:
        raise ValueError("non-loopback server binding needs --allow-network")
    if args.max_request_chars < 20:
        raise ValueError("server max_request_chars must be at least 20")
    if args.speed <= 0:
        raise ValueError("reader speed must be greater than zero")
    if args.sample_rate not in {24_000, 48_000}:
        raise ValueError("reader sample_rate must be 24000 or 48000")
    if not 0 <= args.chunk_pause_ms <= 2_000:
        raise ValueError("reader chunk_pause_ms must be between 0 and 2000")
    args.pronunciations = _pronunciations(config)


def _apply_reader_config(args: argparse.Namespace) -> None:
    """Apply viewer-only settings after the shared local server configuration."""
    config = load_config(args.config)
    if args.theme is None:
        args.theme = config_value(config, "viewer", "theme", "kuro-nezumi")
    if args.theme_css is None and args.theme == "custom":
        configured_css = config_value(config, "viewer", "custom_css", None)
        if configured_css is not None:
            if not isinstance(configured_css, str) or not configured_css.strip():
                raise ValueError("viewer custom_css must be a non-empty path")
            args.theme_css = Path(configured_css)
            if args.config is not None and not args.theme_css.is_absolute():
                args.theme_css = args.config.parent / args.theme_css
    if args.theme not in {"kuro-nezumi", "default", "custom"}:
        raise ValueError("viewer theme must be kuro-nezumi, default, or custom")
    if args.theme == "custom" and args.theme_css is None:
        raise ValueError("--theme custom needs --theme-css PATH or [viewer] custom_css")
    if args.theme_css is not None and args.theme != "custom":
        raise ValueError("--theme-css is only used with --theme custom")
    if args.theme_css is not None and not args.theme_css.is_file():
        raise ValueError(f"custom theme CSS not found: {args.theme_css}")


def _pronunciations(config: dict[str, dict[str, object]]) -> tuple[tuple[str, str], ...]:
    values = config.get("pronunciation", {})
    replacements: list[tuple[str, str]] = []
    for source, spoken in values.items():
        if not isinstance(source, str) or not isinstance(spoken, str) or not source.strip() or not spoken.strip():
            raise ValueError("[pronunciation] entries must map non-empty text to non-empty spoken text")
        replacements.append((source, spoken))
    return tuple(replacements)


def _apply_thread_limits(resources: ResourceSettings) -> None:
    # These influence CPU-side phonemization/BLAS when those libraries honour them.
    for key in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(resources.cpu_threads)
    # Prevent tokenizers from creating an unbounded second worker pool beside
    # the controlled extraction/cleanup workers.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _print_inspection(path: Path) -> None:
    with PDFDocument(path) as document:
        _print_document_inspection(document)


def _print_document_inspection(document: PDFDocument) -> None:
    page_count, chapters = document.inspect()
    print(f"pages: {page_count}")
    if not chapters:
        print("outline: none (use --pages START-END)")
        return
    print("outline:")
    for chapter in chapters:
        print(f"  {chapter.number:>3}  pp. {chapter.start_page}-{chapter.end_page}  {chapter.title}")


def _run_book(args: argparse.Namespace) -> int:
    if not args.pdf.is_file():
        raise ValueError(f"PDF not found: {args.pdf}")
    if args.list or args.metadata:
        with PDFDocument(args.pdf) as document:
            if args.list:
                _print_document_inspection(document)
            if args.metadata:
                print(json.dumps(document.metadata(), indent=2))
        return 0
    _apply_book_config(args)
    resources = _resources(args)
    _apply_thread_limits(resources)
    cleanup = CleanupOptions(
        code=args.code, schemas=args.schemas, tables=args.tables, urls=args.urls, citations=args.citations,
        footnotes=args.footnotes, headers_footers=not args.keep_headers_footers,
    )
    options = RenderOptions(
        voice=args.voice, speed=args.speed, sample_rate=args.sample_rate, chunk_pause_ms=args.chunk_pause_ms,
        pronunciations=args.pronunciations, audio_format=args.format, resume=args.resume,
    )
    summary_options = SummaryOptions(
        model=args.summary_model or SummaryOptions().model,
        max_context_chars=args.summary_context_chars,
        max_output_tokens=args.summary_max_tokens,
        max_reference_chapters=args.summary_references,
    )
    podcast_options = PodcastOptions(
        max_output_tokens=args.podcast_max_tokens,
        max_turns=args.podcast_max_turns,
    )
    with PDFDocument(args.pdf) as document:
        page_count, chapters = document.inspect()
        book_metadata = document.metadata()
        if args.chapter:
            pages, label = chapter_pages(chapters, args.chapter)
            jobs = [(pages, label)]
        elif args.chapters:
            # A chapter range deliberately remains a set of independent outputs so
            # one successful chapter is never held hostage by a later failure.
            range_pages, _ = chapter_pages(chapters, args.chapters)
            first, last = range_pages[0], range_pages[-1]
            jobs = [
                (list(range(chapter.start_page, chapter.end_page + 1)), f"chapter-{chapter.number:02d}")
                for chapter in chapters
                if first <= chapter.start_page and chapter.end_page <= last
            ]
        elif args.pages:
            jobs = [(page_range(args.pages, page_count), f"pages-{args.pages}")]
        else:
            jobs = (
                [(list(range(chapter.start_page, chapter.end_page + 1)), f"chapter-{chapter.number:02d}") for chapter in chapters]
                if chapters
                else [(list(range(1, page_count + 1)), "whole-book")]
            )
        # Keep selected text in memory once. This is small compared with model
        # weights and lets summary generation and rendering share extraction.
        # The callback is intentionally inside PDFDocument's single-reader
        # extraction loop, avoiding a second parse merely to show progress.
        pages_total = sum(len(pages) for pages, _ in jobs)
        pages_done = 0
        extraction_started = time.perf_counter()
        extraction_progress = TerminalProgress("PDF")
        job_pages = []
        for pages, label in jobs:
            def on_page(page: int, within_job: int, _job_total: int) -> None:
                extraction_progress.update(
                    ProgressEvent(
                        "extracting",
                        pages_done + within_job,
                        pages_total,
                        pages_done + within_job,
                        elapsed_seconds=time.perf_counter() - extraction_started,
                        detail=f"page {page} ({label})",
                    )
                )

            extracted = document.extract_pages(pages, progress=on_page)
            pages_done += len(pages)
            job_pages.append((pages, label, extracted))
        extraction_progress.finish(
            ProgressEvent(
                "extracted",
                pages_total,
                pages_total,
                pages_total,
                elapsed_seconds=time.perf_counter() - extraction_started,
                detail=f"{pages_total} pages ready",
            )
        )
        if args.translate:
            return _run_farsi_translation(args, job_pages, cleanup, resources, book_metadata)
        summaries = {}
        podcasts = {}
        if args.summarize or args.podcast:
            summary_backend = MLXSummaryBackend(summary_options)
            summary_backend.configure(resources)
            analysis_total = len(job_pages) * int(args.summarize) + len(job_pages) * int(args.podcast)
            analysis_done = 0
            analysis_started = time.perf_counter()
            analysis_progress = TerminalProgress("Local analysis")
            try:
                for pages, label, extracted in job_pages:
                    context = build_summary_context(document, extracted, chapters, cleanup, summary_options)
                    if args.summarize:
                        result = summary_backend.summarize(context)
                        markdown, details = write_summary(args.output_dir, label, context, result)
                        summaries[label] = (result, markdown, details)
                        analysis_done += 1
                        analysis_progress.update(
                            ProgressEvent(
                                "writing conclusion",
                                analysis_done,
                                analysis_total,
                                analysis_done,
                                elapsed_seconds=time.perf_counter() - analysis_started,
                                detail=label,
                            )
                        )
                    if args.podcast:
                        result = summary_backend.podcast(context, podcast_options)
                        markdown, details = write_podcast(
                            args.output_dir,
                            label,
                            context,
                            result,
                            host_voice=args.podcast_host_voice,
                            explainer_voice=args.podcast_explainer_voice,
                        )
                        podcasts[label] = (result, markdown, details)
                        analysis_done += 1
                        analysis_progress.update(
                            ProgressEvent(
                                "writing podcast",
                                analysis_done,
                                analysis_total,
                                analysis_done,
                                elapsed_seconds=time.perf_counter() - analysis_started,
                                detail=label,
                            )
                        )
            finally:
                # The LLM and Kokoro intentionally never stay loaded together.
                summary_backend.close()
                analysis_progress.finish()
        backend = KokoroMLXBackend()
        backend.configure(resources)
        try:
            # The document stays open, so a whole book does not reparse its
            # cross-reference table for every independently resumable chapter.
            reports = []
            for _, label, extracted in job_pages:
                narration_progress = TerminalProgress(label)
                narration = render(
                    extracted,
                    args.output_dir,
                    label,
                    backend,
                    resources,
                    cleanup,
                    options,
                    book_metadata,
                    progress=narration_progress.update,
                )
                narration_progress.finish()
                summary_audio = None
                if label in summaries:
                    result, _, _ = summaries[label]
                    summary_cleanup = CleanupOptions(
                        code="read", schemas="read", tables="read", urls="read", citations="read", footnotes="read",
                        headers_footers=False,
                    )
                    summary_progress = TerminalProgress(f"{label} conclusion")
                    summary_audio = render(
                        [(0, f"Conclusion. {result.text}")], args.output_dir, f"{label}-summary", backend,
                        resources, summary_cleanup, options, book_metadata, progress=summary_progress.update,
                    )
                    summary_progress.finish()
                podcast_audio = None
                if label in podcasts:
                    result, _, _ = podcasts[label]
                    podcast_progress = TerminalProgress(f"{label} podcast")
                    podcast_audio = render_script(
                        [
                            ScriptLine(
                                index + 1,
                                turn.text,
                                args.podcast_host_voice if turn.speaker == "host" else args.podcast_explainer_voice,
                            )
                            for index, turn in enumerate(result.turns)
                        ],
                        args.output_dir,
                        f"{label}-podcast",
                        backend,
                        resources,
                        options,
                        book_metadata,
                        turn_pause_ms=args.podcast_turn_pause_ms,
                        progress=podcast_progress.update,
                    )
                    podcast_progress.finish()
                reports.append((narration, summary_audio, summaries.get(label), podcast_audio, podcasts.get(label)))
        finally:
            backend.close()
    print(json.dumps([
        {
            "output": str(report.output),
            "chunks_total": report.chunks_total,
            "chunks_synthesized": report.chunks_synthesized,
            "audio_seconds": round(report.audio_seconds, 2),
            "wall_seconds": round(report.wall_seconds, 2),
            "detected_code_lines": report.detected_code_lines,
            "detected_schema_lines": report.detected_schema_lines,
            "detected_table_lines": report.detected_table_lines,
            **(
                {
                    "summary": {
                        "text": str(summary_files[1]),
                        "details": str(summary_files[2]),
                        "audio": str(summary_audio.output),
                        "input_tokens": summary_files[0].input_tokens,
                        "output_tokens": summary_files[0].output_tokens,
                        "wall_seconds": round(summary_files[0].wall_seconds, 2),
                        "mlx_active_mb": round(summary_files[0].mlx_active_mb, 1),
                        "mlx_peak_mb": round(summary_files[0].mlx_peak_mb, 1),
                    }
                }
                if summary_files and summary_audio
                else {}
            ),
            **(
                {
                    "podcast": {
                        "transcript": str(podcast_files[1]),
                        "details": str(podcast_files[2]),
                        "audio": str(podcast_audio.output),
                        "turns": len(podcast_files[0].turns),
                        "chunks_total": podcast_audio.chunks_total,
                        "chunks_synthesized": podcast_audio.chunks_synthesized,
                        "audio_seconds": round(podcast_audio.audio_seconds, 2),
                        "tts_wall_seconds": round(podcast_audio.wall_seconds, 2),
                        "input_tokens": podcast_files[0].generation.input_tokens,
                        "output_tokens": podcast_files[0].generation.output_tokens,
                        "wall_seconds": round(podcast_files[0].generation.wall_seconds, 2),
                        "mlx_active_mb": round(podcast_files[0].generation.mlx_active_mb, 1),
                        "mlx_peak_mb": round(podcast_files[0].generation.mlx_peak_mb, 1),
                    }
                }
                if podcast_files and podcast_audio
                else {}
            ),
        }
        for report, summary_audio, summary_files, podcast_audio, podcast_files in reports
    ], indent=2))
    return 0


def _run_farsi_translation(
    args: argparse.Namespace,
    job_pages,
    cleanup: CleanupOptions,
    resources: ResourceSettings,
    book_metadata: dict[str, object],
) -> int:
    """Translate with MLX first, unload it, then narrate in native Persian."""
    # Validate the local model up front. A missing voice should not cost a
    # lengthy translation pass before reporting the one-time setup command.
    model_path = piper_voice_path(args.farsi_model_dir, args.farsi_voice)
    translation_options = TranslationOptions(
        model=args.translation_model,
        backend=args.translation_backend,
        max_source_chars=args.translation_chunk_chars,
        max_output_tokens=args.translation_max_tokens,
        protected_terms=args.translation_protected_terms,
        glossary=args.translation_glossary,
        transliterations=args.translation_transliterations,
    )
    translator = (
        NLLBPersianTranslator(args.translation_model)
        if args.translation_backend == "nllb"
        else MLXSummaryBackend(SummaryOptions(model=args.translation_model))
    )
    translator.configure(resources)
    translated_jobs = []
    try:
        for _, label, extracted in job_pages:
            translation_progress = TerminalProgress(f"{label} Persian")
            report = translate_pages_to_persian(
                extracted,
                args.output_dir,
                f"{label}-fa",
                translator,
                resources,
                cleanup,
                translation_options,
                resume=args.resume,
                progress=translation_progress.update,
            )
            translation_progress.finish()
            translated_jobs.append((label, report))
    finally:
        # The Qwen translator must not remain resident while narration starts.
        translator.close()

    backend = PiperFarsiBackend(
        model_path,
        noise_scale=args.farsi_noise_scale,
        noise_w_scale=args.farsi_noise_w_scale,
    )
    backend.configure(resources)
    farsi_render_options = RenderOptions(
        voice=args.farsi_voice,
        speed=args.speed,
        sample_rate=args.sample_rate,
        chunk_pause_ms=args.chunk_pause_ms,
        # English pronunciation substitutions are for Kokoro source reading;
        # applying them to translated Persian could corrupt intentional terms.
        pronunciations=(),
        audio_format=args.format,
        resume=args.resume,
    )
    try:
        reports = []
        for label, translation in translated_jobs:
            lines = [ScriptLine(segment.page, segment.translation, args.farsi_voice) for segment in translation.segments]
            narration_progress = TerminalProgress(f"{label} Persian audio")
            narration = render_script(
                lines,
                args.output_dir,
                f"{label}-fa",
                backend,
                resources,
                farsi_render_options,
                book_metadata,
                turn_pause_ms=args.chunk_pause_ms,
                script_kind="translation",
                script_metadata={"language": "fa", "translation_sidecar": str(translation.sidecar)},
                progress=narration_progress.update,
            )
            narration_progress.finish()
            reports.append((narration, translation))
    finally:
        backend.close()
    print(json.dumps([
        {
            "output": str(narration.output),
            "chunks_total": narration.chunks_total,
            "chunks_synthesized": narration.chunks_synthesized,
            "audio_seconds": round(narration.audio_seconds, 2),
            "wall_seconds": round(narration.wall_seconds, 2),
            "translation": {
                "language": translation.language,
                "backend": translation.backend,
                "model": translation.model,
                "segments": len(translation.segments),
                "sidecar": str(translation.sidecar),
            },
        }
        for narration, translation in reports
    ], indent=2))
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    resources = _resources(args)
    _apply_thread_limits(resources)
    backend = KokoroMLXBackend()
    try:
        backend.configure(resources)
        progress = TerminalProgress("Benchmark")
        reports, recommended = benchmark_candidates(backend, resources, runs=args.runs, progress=progress.update)
        progress.finish()
    finally:
        backend.close()
    print(json.dumps({
        "backend": backend.name,
        "resources": resources.as_dict(),
        "candidates": [report.as_dict() for report in reports],
        "recommendation": {
            "chunk_chars": recommended.chunk_chars,
            "realtime_factor": recommended.realtime_factor,
            "reason": "highest measured real-time factor within the advisory memory ceiling",
        },
    }, indent=2))
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    _apply_serve_config(args)
    resources = _resources(args)
    _apply_thread_limits(resources)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - install-time concern
        raise RuntimeError("server support is unavailable; run `uv sync --extra server`") from exc
    from .server import ServerSettings, create_app

    options = RenderOptions(
        voice=args.voice,
        speed=args.speed,
        sample_rate=args.sample_rate,
        chunk_pause_ms=args.chunk_pause_ms,
        pronunciations=args.pronunciations,
    )
    app = create_app(
        ServerSettings(
            resources=resources,
            render=options,
            max_request_chars=args.max_request_chars,
            allowed_origins=tuple(args.allow_origin),
        )
    )
    progress = TerminalProgress("Local TTS server")
    progress.finish(ProgressEvent("ready", 1, 1, 1, elapsed_seconds=0.0, detail=f"port {args.port}"))
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    return 0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pdfjs_root() -> Path:
    return _project_root() / "web" / "node_modules" / "pdfjs-dist"


def _run_setup_viewer(args: argparse.Namespace) -> int:
    web_root = _project_root() / "web"
    if not (web_root / "package.json").is_file():
        raise RuntimeError("local PDF.js setup metadata is missing; reinstall the project")
    if shutil.which("npm") is None:
        raise RuntimeError("Node.js and npm are required once for the local PDF.js viewer setup")
    if not args.force and (_pdfjs_root() / "build" / "pdf.mjs").is_file():
        print(f"PDF.js is already installed locally at {_pdfjs_root()}")
        return 0
    progress = TerminalProgress("PDF.js setup")
    started = time.perf_counter()
    progress.update(ProgressEvent("installing", 0, 1, 0, elapsed_seconds=0.0, detail="downloading local assets"))
    subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=web_root, check=True)
    progress.finish(ProgressEvent("ready", 1, 1, 1, elapsed_seconds=time.perf_counter() - started, detail="local assets installed"))
    print(f"PDF.js is ready locally at {_pdfjs_root()}")
    return 0


def _run_setup_farsi(args: argparse.Namespace) -> int:
    try:
        import piper  # noqa: F401
    except ImportError as exc:  # pragma: no cover - install-time concern
        raise RuntimeError("Persian TTS support is unavailable; run `uv sync --extra farsi`") from exc
    args.model_dir.mkdir(parents=True, exist_ok=True)
    progress = TerminalProgress("Persian voice setup")
    started = time.perf_counter()
    progress.update(ProgressEvent("downloading", 0, 1, 0, elapsed_seconds=0.0, detail=args.voice))
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", args.voice, "--download-dir", str(args.model_dir)],
        check=True,
    )
    progress.finish(ProgressEvent("ready", 1, 1, 1, elapsed_seconds=time.perf_counter() - started, detail=args.voice))
    print(f"Persian voice {args.voice} is ready locally in {args.model_dir}")
    return 0


def _run_read(args: argparse.Namespace) -> int:
    if not args.pdf.is_file():
        raise ValueError(f"PDF not found: {args.pdf}")
    _apply_serve_config(args)
    _apply_reader_config(args)
    resources = _resources(args)
    _apply_thread_limits(resources)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - install-time concern
        raise RuntimeError("server support is unavailable; run `uv sync --extra server`") from exc
    from .reader import create_reader_app, viewer_address
    from .server import ServerSettings

    options = RenderOptions(
        voice=args.voice,
        speed=args.speed,
        sample_rate=args.sample_rate,
        chunk_pause_ms=args.chunk_pause_ms,
        pronunciations=args.pronunciations,
    )
    viewer_url = viewer_address(
        args.host,
        args.port,
        args.pdf,
        theme=args.theme,
        custom_theme=args.theme_css is not None,
    )
    app = create_reader_app(
        ServerSettings(
            resources=resources,
            render=options,
            max_request_chars=args.max_request_chars,
            allowed_origins=tuple(args.allow_origin),
        ),
        args.pdf,
        _pdfjs_root(),
        _project_root() / "web" / "reader",
        viewer_url,
        open_browser=not args.no_open,
        custom_css=args.theme_css,
    )
    progress = TerminalProgress("Local reader")
    progress.finish(ProgressEvent("ready", 1, 1, 1, elapsed_seconds=0.0, detail=f"port {args.port}"))
    print(f"Opening local reader: {viewer_url}")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    # Support the requested ergonomic form: local-tts book.pdf --pages 1-2.
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if argv and argv[0] not in {"book", "benchmark", "serve", "setup-viewer", "setup-farsi", "read", "-h", "--help"}:
        argv = ["book", *argv]
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            return _run_benchmark(args)
        if args.command == "serve":
            return _run_serve(args)
        if args.command == "setup-viewer":
            return _run_setup_viewer(args)
        if args.command == "setup-farsi":
            return _run_setup_farsi(args)
        if args.command == "read":
            return _run_read(args)
        if args.command == "book":
            return _run_book(args)
        parser.print_help()
        return 2
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
