"""Command-line interface for PDF selection, rendering, and benchmarking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from .backends import KokoroMLXBackend
from .benchmark import benchmark_candidates
from .config import config_value, load_config
from .models import CleanupOptions, PROFILES, ProfileName, RenderOptions, ResourceSettings
from .pdf import PDFDocument, chapter_pages, page_range
from .render import ScriptLine, render, render_script
from .summarize import (
    MLXSummaryBackend,
    PodcastOptions,
    SummaryOptions,
    build_summary_context,
    write_podcast,
    write_summary,
)


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
        job_pages = [(pages, label, document.extract_pages(pages)) for pages, label in jobs]
        summaries = {}
        podcasts = {}
        if args.summarize or args.podcast:
            summary_backend = MLXSummaryBackend(summary_options)
            summary_backend.configure(resources)
            try:
                for pages, label, extracted in job_pages:
                    context = build_summary_context(document, extracted, chapters, cleanup, summary_options)
                    if args.summarize:
                        result = summary_backend.summarize(context)
                        markdown, details = write_summary(args.output_dir, label, context, result)
                        summaries[label] = (result, markdown, details)
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
            finally:
                # The LLM and Kokoro intentionally never stay loaded together.
                summary_backend.close()
        backend = KokoroMLXBackend()
        backend.configure(resources)
        try:
            # The document stays open, so a whole book does not reparse its
            # cross-reference table for every independently resumable chapter.
            reports = []
            for _, label, extracted in job_pages:
                narration = render(extracted, args.output_dir, label, backend, resources, cleanup, options, book_metadata)
                summary_audio = None
                if label in summaries:
                    result, _, _ = summaries[label]
                    summary_cleanup = CleanupOptions(
                        code="read", schemas="read", tables="read", urls="read", citations="read", footnotes="read",
                        headers_footers=False,
                    )
                    summary_audio = render(
                        [(0, f"Conclusion. {result.text}")], args.output_dir, f"{label}-summary", backend,
                        resources, summary_cleanup, options, book_metadata,
                    )
                podcast_audio = None
                if label in podcasts:
                    result, _, _ = podcasts[label]
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
                    )
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


def _run_benchmark(args: argparse.Namespace) -> int:
    resources = _resources(args)
    _apply_thread_limits(resources)
    backend = KokoroMLXBackend()
    try:
        backend.configure(resources)
        reports, recommended = benchmark_candidates(backend, resources, runs=args.runs)
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
    subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=web_root, check=True)
    print(f"PDF.js is ready locally at {_pdfjs_root()}")
    return 0


def _run_read(args: argparse.Namespace) -> int:
    if not args.pdf.is_file():
        raise ValueError(f"PDF not found: {args.pdf}")
    _apply_serve_config(args)
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
    viewer_url = viewer_address(args.host, args.port, args.pdf)
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
    )
    print(f"Opening local reader: {viewer_url}")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    # Support the requested ergonomic form: local-tts book.pdf --pages 1-2.
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if argv and argv[0] not in {"book", "benchmark", "serve", "setup-viewer", "read", "-h", "--help"}:
        argv = ["book", *argv]
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            return _run_benchmark(args)
        if args.command == "serve":
            return _run_serve(args)
        if args.command == "setup-viewer":
            return _run_setup_viewer(args)
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
