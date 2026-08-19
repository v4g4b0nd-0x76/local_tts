"""Loopback HTTP streaming API for local readers and PDF.js integrations."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import json
import re
import struct
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import partial

import anyio
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .backends import KokoroMLXBackend
from .backends.base import TTSBackend
from .models import RenderOptions, ResourceSettings
from .text import apply_pronunciations


@dataclass(frozen=True)
class ServerSettings:
    """Local server defaults. Binding remains a CLI/uvicorn concern."""

    resources: ResourceSettings
    render: RenderOptions = field(default_factory=RenderOptions)
    max_request_chars: int = 50_000
    allowed_origins: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SpeechParameters:
    text: str
    voice: str
    speed: float
    sample_rate: int
    chunk_chars: int


@dataclass(frozen=True)
class _TextSegment:
    index: int
    start: int
    end: int
    text: str


@dataclass
class _ServerState:
    settings: ServerSettings
    backend: TTSBackend
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SpeechRequest(BaseModel):
    """A bounded text request. All audio is generated on this machine."""

    text: str
    voice: str | None = None
    speed: float | None = None
    sample_rate: int | None = None
    chunk_chars: int | None = None


def create_app(
    settings: ServerSettings,
    backend: TTSBackend | None = None,
    startup_callback: Callable[[], None] | None = None,
) -> FastAPI:
    """Build an injectable app, keeping the normal CLI free of HTTP imports."""
    if settings.max_request_chars < 20:
        raise ValueError("server max_request_chars must be at least 20")
    state = _ServerState(settings, backend or KokoroMLXBackend())
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.backend.configure(settings.resources)
        if startup_callback:
            startup_callback()
        try:
            yield
        finally:
            state.backend.close()

    app = FastAPI(title="local-tts", version="0.1.0", docs_url=None, redoc_url=None, lifespan=lifespan)
    # `null` allows a locally opened PDF.js viewer; the regex permits standard
    # loopback dev servers. The service itself should stay bound to loopback.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["null", *settings.allowed_origins],
        allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.get("/healthz")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "backend": state.backend.name,
            "stream_format": "ndjson-pcm_s16le",
            "defaults": {
                "voice": settings.render.voice,
                "speed": settings.render.speed,
                "sample_rate": settings.render.sample_rate,
                "chunk_chars": settings.resources.chunk_chars,
            },
        }

    @app.post("/v1/speech/stream")
    async def speech_stream(payload: SpeechRequest, request: Request) -> StreamingResponse:
        parameters = _speech_parameters(payload, settings)
        return StreamingResponse(
            _ndjson_audio_events(request, state, parameters),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.post("/v1/speech/wav")
    async def speech_wav(payload: SpeechRequest, request: Request) -> StreamingResponse:
        parameters = _speech_parameters(payload, settings)
        return StreamingResponse(
            _wav_audio_stream(request, state, parameters),
            media_type="audio/wav",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    app.state.local_tts = state
    return app


def _speech_parameters(payload: SpeechRequest, settings: ServerSettings) -> _SpeechParameters:
    text = payload.text
    if not text.strip():
        raise HTTPException(status_code=422, detail="text must contain readable characters")
    if len(text) > settings.max_request_chars:
        raise HTTPException(
            status_code=413,
            detail=f"text exceeds the local server limit of {settings.max_request_chars} characters",
        )
    voice = (payload.voice or settings.render.voice).strip()
    speed = payload.speed if payload.speed is not None else settings.render.speed
    sample_rate = payload.sample_rate if payload.sample_rate is not None else settings.render.sample_rate
    chunk_chars = payload.chunk_chars if payload.chunk_chars is not None else settings.resources.chunk_chars
    if not voice:
        raise HTTPException(status_code=422, detail="voice must be non-empty")
    if speed <= 0:
        raise HTTPException(status_code=422, detail="speed must be greater than zero")
    if sample_rate not in {24_000, 48_000}:
        raise HTTPException(status_code=422, detail="sample_rate must be 24000 or 48000")
    if not 20 <= chunk_chars <= 4_000:
        raise HTTPException(status_code=422, detail="chunk_chars must be between 20 and 4000")
    return _SpeechParameters(text, voice, speed, sample_rate, chunk_chars)


async def _synthesized_segments(
    request: Request,
    state: _ServerState,
    parameters: _SpeechParameters,
    segments: list[_TextSegment],
) -> AsyncIterator[tuple[_TextSegment, bytes, int]]:
    """Keep one MLX inference stream, releasing the event loop between chunks."""
    pronunciations = state.settings.render.pronunciations
    async with state.lock:
        for segment in segments:
            if await request.is_disconnected():
                return
            spoken_text = apply_pronunciations(segment.text, pronunciations)
            result = await anyio.to_thread.run_sync(
                partial(
                    state.backend.synthesize,
                    spoken_text,
                    voice=parameters.voice,
                    speed=parameters.speed,
                    sample_rate=parameters.sample_rate,
                )
            )
            yield segment, _pcm_s16le(result.audio, result.sample_rate, state.settings.render.chunk_pause_ms), result.sample_rate


async def _ndjson_audio_events(
    request: Request, state: _ServerState, parameters: _SpeechParameters
) -> AsyncIterator[bytes]:
    request_id = uuid.uuid4().hex
    segments = _segments(parameters.text, parameters.chunk_chars)
    yield _ndjson({
        "type": "start",
        "request_id": request_id,
        "format": "pcm_s16le",
        "channels": 1,
        "sample_rate": parameters.sample_rate,
        "chunks_total": len(segments),
    })
    try:
        count = 0
        async for segment, pcm, sample_rate in _synthesized_segments(request, state, parameters, segments):
            count += 1
            yield _ndjson({
                "type": "audio",
                "request_id": request_id,
                "sequence": segment.index,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "sample_rate": sample_rate,
                "audio": base64.b64encode(pcm).decode("ascii"),
            })
        yield _ndjson({"type": "done", "request_id": request_id, "chunks": count})
    except Exception:
        # Once an HTTP stream begins its status cannot change. The browser
        # bridge handles this structured terminal event without exposing local
        # paths or implementation details.
        yield _ndjson({"type": "error", "request_id": request_id, "message": "local synthesis failed"})


async def _wav_audio_stream(
    request: Request, state: _ServerState, parameters: _SpeechParameters
) -> AsyncIterator[bytes]:
    # A sentinel data size makes this a progressive WAV stream. Consumers that
    # require a finalized RIFF length should use the NDJSON endpoint instead.
    yield _wav_header(parameters.sample_rate)
    segments = _segments(parameters.text, parameters.chunk_chars)
    async for _, pcm, _ in _synthesized_segments(request, state, parameters, segments):
        yield pcm


def _segments(text: str, max_chars: int) -> list[_TextSegment]:
    """Sentence-biased source slices whose offsets remain exact for highlights."""
    segments: list[_TextSegment] = []
    cursor = 0
    index = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        end_limit = min(len(text), cursor + max_chars)
        if end_limit == len(text):
            end = len(text)
        else:
            window = text[cursor:end_limit]
            sentence_breaks = [match.end() for match in re.finditer(r"[.!?](?=\s)", window)]
            if sentence_breaks:
                end = cursor + sentence_breaks[-1]
            else:
                whitespace = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
                end = cursor + whitespace if whitespace >= max_chars // 2 else end_limit
        while end > cursor and text[end - 1].isspace():
            end -= 1
        if end <= cursor:
            end = min(len(text), cursor + max_chars)
        segments.append(_TextSegment(index, cursor, end, text[cursor:end]))
        index += 1
        cursor = end
    return segments


def _pcm_s16le(audio: np.ndarray, sample_rate: int, pause_ms: int) -> bytes:
    samples = np.asarray(audio, dtype=np.float32)
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2", copy=False).tobytes()
    return pcm + (b"\0" * (round(sample_rate * pause_ms / 1000) * 2))


def _wav_header(sample_rate: int) -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        0xFFFFFFFF,
    )


def _ndjson(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
