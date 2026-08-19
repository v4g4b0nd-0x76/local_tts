from io import StringIO

from local_tts.progress import ProgressEvent, TerminalProgress


def test_terminal_progress_writes_a_bar_and_realtime_rate_to_stderr_stream() -> None:
    stream = StringIO()
    progress = TerminalProgress("Narration", stream=stream, width=10)
    progress.update(
        ProgressEvent(
            "synthesizing",
            completed=1,
            total=2,
            units_completed=3,
            audio_seconds=12.0,
            elapsed_seconds=4.0,
            detail="page 4",
        )
    )
    progress.finish(
        ProgressEvent(
            "complete",
            completed=2,
            total=2,
            units_completed=4,
            audio_seconds=16.0,
            elapsed_seconds=4.0,
        )
    )

    rendered = stream.getvalue()
    assert "[#####-----]  50.0%" in rendered
    assert "3.00x realtime" in rendered
    assert "complete" in rendered
