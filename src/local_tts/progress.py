"""Small dependency-free progress reporting for local-tts CLI workloads."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO


@dataclass(frozen=True)
class ProgressEvent:
    """A unit of observable work emitted by extraction, generation, or encoding."""

    phase: str
    completed: float
    total: float
    units_completed: int
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    detail: str = ""


class TerminalProgress:
    """Render a compact progress bar on stderr without contaminating JSON stdout."""

    def __init__(self, label: str, *, stream: TextIO | None = None, width: int = 24) -> None:
        self.label = label
        self.stream = stream or sys.stderr
        self.width = width
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_phase = ""
        self._last_rendered = 0.0
        self._ended = False

    def update(self, event: ProgressEvent) -> None:
        """Show progress, phase, and current generation rate when it is known."""
        if self._ended:
            return
        now = time.perf_counter()
        final = event.total > 0 and event.completed >= event.total
        # A TTY is refreshed frequently enough to feel live. When stderr is
        # redirected, emit bounded complete lines instead of one per chunk.
        if not final and event.phase == self._last_phase and now - self._last_rendered < 0.12:
            return
        if not self._interactive and not final and event.phase == self._last_phase and now - self._last_rendered < 0.75:
            return
        self._write(self._format(event), newline=not self._interactive)
        self._last_phase = event.phase
        self._last_rendered = now

    def finish(self, event: ProgressEvent | None = None) -> None:
        """Finish the current line, optionally rendering a final complete event."""
        if self._ended:
            return
        if event is not None:
            self.update(event)
        if self._interactive:
            self.stream.write("\n")
            self.stream.flush()
        self._ended = True

    def _format(self, event: ProgressEvent) -> str:
        if event.total > 0:
            fraction = min(1.0, max(0.0, event.completed / event.total))
            filled = round(self.width * fraction)
            bar = "#" * filled + "-" * (self.width - filled)
            progress = f"[{bar}] {fraction:>6.1%}"
        else:
            progress = "[working]"
        rate = _rate(event)
        detail = f" | {event.detail}" if event.detail else ""
        return f"{self.label}: {event.phase} {progress} | {rate}{detail}"

    def _write(self, text: str, *, newline: bool) -> None:
        prefix = "" if newline else "\r"
        suffix = "\n" if newline else ""
        self.stream.write(f"{prefix}{text}{suffix}")
        self.stream.flush()


def _rate(event: ProgressEvent) -> str:
    if event.elapsed_seconds <= 0:
        return "starting"
    if event.audio_seconds > 0:
        return f"{event.audio_seconds / event.elapsed_seconds:.2f}x realtime"
    return f"{event.units_completed / event.elapsed_seconds:.2f} items/s"
