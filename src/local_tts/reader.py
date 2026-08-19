"""Self-contained local PDF.js reader server for one user-selected PDF."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .backends.base import TTSBackend
from .server import ServerSettings, create_app


def create_reader_app(
    settings: ServerSettings,
    pdf_path: Path,
    pdfjs_root: Path,
    reader_root: Path,
    viewer_url: str,
    *,
    open_browser: bool,
    backend: TTSBackend | None = None,
    custom_css: Path | None = None,
) -> FastAPI:
    """Serve only the chosen PDF, local PDF.js assets, and the loopback TTS API."""
    source = pdf_path.resolve()
    if not source.is_file():
        raise ValueError(f"PDF not found: {pdf_path}")
    _require_pdfjs_assets(pdfjs_root)
    if not (reader_root / "index.html").is_file():
        raise ValueError("local reader assets are missing; reinstall the project")
    bridge = reader_root.parent / "pdfjs-local-tts.js"
    if not bridge.is_file():
        raise ValueError("local PDF.js speech bridge is missing; reinstall the project")
    if custom_css is not None:
        custom_css = custom_css.resolve()
        if not custom_css.is_file():
            raise ValueError(f"custom theme CSS not found: {custom_css}")

    app = create_app(
        settings,
        backend,
        startup_callback=(lambda: webbrowser.open(viewer_url, new=2)) if open_browser else None,
    )
    @app.get("/local-tts/pdfjs-local-tts.js", include_in_schema=False)
    async def speech_bridge() -> FileResponse:
        return FileResponse(bridge, media_type="text/javascript")

    if custom_css is not None:
        @app.get("/reader/custom-theme.css", include_in_schema=False)
        async def custom_theme() -> FileResponse:
            return FileResponse(custom_css, media_type="text/css")

    @app.get("/file", include_in_schema=False)
    async def selected_file() -> FileResponse:
        return FileResponse(source, media_type="application/pdf", filename=source.name)

    # Routes are registered before the static reader mount so the optional
    # custom stylesheet cannot be swallowed by StaticFiles as a 404.
    app.mount("/assets/pdfjs", StaticFiles(directory=pdfjs_root), name="pdfjs-assets")
    app.mount("/reader", StaticFiles(directory=reader_root, html=True), name="reader-assets")

    return app


def viewer_address(
    host: str,
    port: int,
    pdf_path: Path,
    *,
    theme: str = "kuro-nezumi",
    custom_theme: bool = False,
) -> str:
    """Construct a loopback URL without disclosing the source filesystem path."""
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    custom = "&custom_theme=1" if custom_theme else ""
    return f"http://{display_host}:{port}/reader/?file=/file&name={quote(pdf_path.name)}&theme={quote(theme)}{custom}"


def _require_pdfjs_assets(pdfjs_root: Path) -> None:
    required = ("build/pdf.mjs", "build/pdf.worker.mjs", "web/pdf_viewer.mjs", "web/pdf_viewer.css")
    missing = [entry for entry in required if not (pdfjs_root / entry).is_file()]
    if missing:
        raise ValueError("PDF.js viewer is not set up; run `local-tts setup-viewer`")
