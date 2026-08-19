# Local concluded summaries

Use `--summarize` (or the tolerated spelling `--summerize`) to create a
conclusion for every selected chapter or page range. The command writes three
independent artifacts beside the normal narration:

- `label-summary.md`: readable conclusion.
- `label-summary.json`: source scope, included chapter context, model metrics,
  and output metadata.
- `label-summary.m4b` (or the selected format): the same conclusion narrated
  with the normal Kokoro voice configuration.

```bash
uv sync --extra summarize
uv run local-tts book.pdf --config config/natural-explanatory-reader.toml \
  --pages 120-160 --summarize
```

## Context policy

The selected pages are always the primary evidence. When the PDF has an
outline, the planner also adds a bounded excerpt from the immediately previous
chapter and from up to three explicitly cited references written as `Chapter
N` in the selected source text. `N` is the one-based outline entry number used
by `local-tts --list`; it never guesses a link from a vague matching title. The
sidecar JSON records exactly which pages were included.

`context_chars`, `max_output_tokens`, and `max_references` in `[summary]` cap
the task. They default to 32,000 source characters, 480 generated tokens, and
three references. Lower `context_chars` first if unified-memory pressure is a
concern.

## Local model and cost

The default is `mlx-community/Qwen3-4B-Instruct-2507-4bit`: an Apple-Silicon
MLX conversion of Qwen's 4B non-thinking instruction model. Its download is
about 2.26 GB and it is Apache-2.0. It is a better fit for an explanatory
conclusion than an ultra-small model, while remaining practical on the 24 GB
M4 Pro. The summary JSON reports actual input/output token counts, wall time,
and MLX active/peak memory on this machine.

The LLM and Kokoro deliberately run in separate phases. The summary model is
released and MLX's cache is cleared before TTS loads, so their model weights
and attention cache are never resident together. A longer context or conclusion
costs more prefill/decode time and memory, but does not slow the normal TTS
inference stream once it starts.

On this project's M4 Pro smoke run, a cached-model summary with 1,101 input
tokens and a 160-token conclusion took 4.04 seconds, used 2.16 GiB active MLX
memory, and reached 3.06 GiB peak MLX memory. Treat that as a reference point,
not a guarantee: the emitted JSON is the measurement for each real book and
setting. The first model download is separate from that timing.
