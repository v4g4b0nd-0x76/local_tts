# PDF layout and book metadata

For every selected page, `local-tts` performs normal text extraction and a
layout-preserving `pypdf` extraction. The latter retains enough horizontal
spacing to identify high-confidence technical blocks without a second PDF
engine or an OCR/cloud service.

- Code detection recognises source syntax, C-style declarations/comments,
  shell prompts, and indented continuations.
- SQL schema starts and their indented fields remain a single `schema` block,
  so the `schemas` policy is respected instead of falling back to `code`.
- Table detection requires a consecutive run of pipe-, tab-, or consistently
  aligned rows. Dotted table-of-contents leaders are explicitly excluded.
- If layout information is missing or ambiguous, cleanup falls back to the
  conservative text rules. It will not invent a technical explanation.

Use `--metadata` to inspect the source PDF without starting TTS:

```bash
uv run local-tts book.pdf --metadata
```

The JSON includes title, author, subject, creator/producer, dates, attachments,
encryption state, page count, and usable outline-entry count. On a render, this
metadata and a per-job layout summary are preserved in
`.label.local-tts/manifest.json`. Non-empty title, author, and subject are also
written to the final `m4a`, `m4b`, or `mp3` tags.

The additional layout pass costs some CPU PDF-extraction time, but it uses no
extra AI model, network request, or accelerator memory. It is intentionally
kept outside Kokoro inference, whose MLX resource profile remains unchanged.
