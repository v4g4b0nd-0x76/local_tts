import base64
import json

import anyio
import httpx
import numpy as np

from local_tts.backends.base import TTSResult
from local_tts.models import RenderOptions, ResourceSettings
from local_tts.reader import create_reader_app, viewer_address
from local_tts.server import ServerSettings, create_app


class FakeBackend:
    name = "fake-server"

    def __init__(self) -> None:
        self.configured = False
        self.closed = False
        self.calls: list[tuple[str, str, float, int]] = []

    def configure(self, resources: ResourceSettings) -> None:
        self.configured = True

    def synthesize(self, text: str, *, voice: str, speed: float, sample_rate: int) -> TTSResult:
        self.calls.append((text, voice, speed, sample_rate))
        return TTSResult(np.array([0.0, 0.25, -0.25], dtype=np.float32), sample_rate)

    def close(self) -> None:
        self.closed = True


def _app() -> tuple[object, FakeBackend]:
    backend = FakeBackend()
    app = create_app(
        ServerSettings(
            resources=ResourceSettings(cpu_threads=1, prefetch=1, chunk_chars=22, memory_gb=1),
            render=RenderOptions(voice="af_bella", speed=0.94, sample_rate=24_000),
        ),
        backend,  # type: ignore[arg-type]
    )
    return app, backend


def _post(app: object, path: str, payload: dict[str, object], headers: dict[str, str] | None = None) -> httpx.Response:
    async def send() -> httpx.Response:
        async with app.router.lifespan_context(app):  # type: ignore[attr-defined]
            transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post(path, json=payload, headers=headers)

    return anyio.run(send)


def test_stream_returns_audio_events_with_source_offsets() -> None:
    app, backend = _app()
    text = "First short sentence. Second short sentence."
    response = _post(
        app,
        "/v1/speech/stream",
        {"text": text, "chunk_chars": 22},
        {"origin": "http://127.0.0.1:8080"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:8080"
    events = [json.loads(line) for line in response.text.splitlines()]
    audio_events = [event for event in events if event["type"] == "audio"]
    assert events[0]["type"] == "start"
    assert events[0]["chunks_total"] == 2
    assert events[-1]["type"] == "done"
    assert len(audio_events) == 2
    assert [event["text"] for event in audio_events] == [text[event["start"]:event["end"]] for event in audio_events]
    assert all(len(base64.b64decode(event["audio"])) == 6 for event in audio_events)
    assert [call[1] for call in backend.calls] == ["af_bella", "af_bella"]
    assert backend.closed is True


def test_wav_endpoint_streams_a_progressive_header_and_pcm() -> None:
    app, backend = _app()
    response = _post(app, "/v1/speech/wav", {"text": "A tiny request."})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"
    assert len(response.content) == 44 + 6
    assert len(backend.calls) == 1


def test_server_rejects_oversized_text_before_synthesis() -> None:
    app, backend = _app()
    response = _post(app, "/v1/speech/stream", {"text": "x" * 50_001})

    assert response.status_code == 413
    assert backend.calls == []


def test_reader_serves_only_the_selected_pdf_and_local_assets_and_opens_browser(tmp_path, monkeypatch) -> None:
    source = tmp_path / "a book.pdf"
    source.write_bytes(b"%PDF-local-test")
    pdfjs = tmp_path / "pdfjs"
    for relative in ("build/pdf.mjs", "build/pdf.worker.mjs", "web/pdf_viewer.mjs", "web/pdf_viewer.css"):
        target = pdfjs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("asset")
    reader = tmp_path / "reader"
    reader.mkdir()
    (reader / "index.html").write_text("<title>reader</title>")
    (tmp_path / "pdfjs-local-tts.js").write_text("bridge")
    opened: list[tuple[str, int]] = []
    monkeypatch.setattr("local_tts.reader.webbrowser.open", lambda url, new: opened.append((url, new)))
    backend = FakeBackend()
    address = viewer_address("127.0.0.1", 8765, source)
    app = create_reader_app(
        ServerSettings(ResourceSettings(cpu_threads=1, prefetch=1, chunk_chars=22, memory_gb=1)),
        source,
        pdfjs,
        reader,
        address,
        open_browser=True,
        backend=backend,  # type: ignore[arg-type]
    )

    async def get_assets() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        async with app.router.lifespan_context(app):  # type: ignore[attr-defined]
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return (
                    await client.get("/reader/"),
                    await client.get("/assets/pdfjs/build/pdf.mjs"),
                    await client.get("/local-tts/pdfjs-local-tts.js"),
                    await client.get("/file"),
                )

    page, asset, bridge, file_response = anyio.run(get_assets)
    assert page.status_code == asset.status_code == bridge.status_code == file_response.status_code == 200
    assert file_response.content == b"%PDF-local-test"
    assert opened == [(address, 2)]
    assert "a%20book.pdf" in address
    assert str(tmp_path) not in address
    assert backend.closed is True
