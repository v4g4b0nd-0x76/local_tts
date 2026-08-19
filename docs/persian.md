# Local Persian translation and narration

This mode translates the selected English PDF text locally, writes the Persian
translation for review, then narrates it with a native Persian Piper voice. It
does not contact ElevenLabs or any other API.

## One-time setup

Install the local translation and Persian-narration runtimes, then download a
voice into the local application data directory:

```bash
cd ~/projects/local_tts
uv sync --extra farsi
uv run local-tts setup-farsi --voice fa_IR-ganji_adabi-medium
```

The default voice is `fa_IR-ganji_adabi-medium`, a small, local baseline. It
is useful for testing the pipeline, but it is not presented as a natural,
premium Iranian-Persian reader. If its accent sounds artificial, that is a
voice-model limitation, not a setting this project can honestly tune away.
Audition at least one alternative before a long book:

```bash
uv run local-tts setup-farsi --voice fa_IR-ganji-medium
uv run local-tts setup-farsi --voice fa_IR-gyro-medium
```

The supplied config uses an intentionally slower `speed = 0.93` and explicit
Piper `noise_scale = 0.45` / `noise_w_scale = 0.65`. These reduce variation and
can make cadence steadier; they do not change the voice's learned Iranian
accent or make it bilingual. The matching command-line overrides are
`--farsi-noise-scale` and `--farsi-noise-w-scale`.

## Translate and listen

The supplied preset enables translation to standard Iranian Persian, supplies a
small technical glossary, and slows narration slightly for a smoother tone:

```bash
uv run local-tts book.pdf --config config/farsi-study-reader.toml --pages 120-160
```

Equivalent explicit form:

```bash
uv run local-tts book.pdf --pages 120-160 --translate fa \
  --farsi-voice fa_IR-ganji_adabi-medium --format m4b
```

The default `nllb` backend is Meta's dedicated NLLB-200 600M translation model;
it is more literal and reliable for source preservation than a chat model in
our local comparison. It downloads once to the local Hugging Face cache, then
runs on Apple MPS. It has a **CC-BY-NC-4.0** model license, suitable for this
personal/non-commercial setup but not a redistribution choice. It is still not
a substitute for a human translator for an important book.

Set `backend = "qwen"` with
`model = "mlx-community/Qwen3-4B-Instruct-2507-4bit"` when a terminology
glossary and more natural rephrasing matter more than strict literalness. Qwen
is also fully local, but the smoke test found that it can mishandle imperfect
PDF/OCR text; review its sidecar especially carefully.

Every output has two review/recovery artifacts:

- `output/pages-120-160-fa.translation.json` contains each English source
  chunk, the Persian translation, selected model, glossary, and generation
  metrics.
- `output/.pages-120-160-fa.local-tts/translation.jsonl` is fsync'd one
  translated chunk at a time. Use `--resume` to reuse the exact Persian wording
  and only synthesize unfinished audio chunks.

The output audio is `pages-120-160-fa.m4b` (or the selected format).

## Language behavior

The translator preserves numbers, units, formulae, qualifications, negation,
and code identifiers; it normalizes Arabic presentation variants such
as `ي`/`ك` into Persian `ی`/`ک` before narration. The Farsi voice therefore
reads Persian sentences and Arabic-script loanwords with a Persian accent—the
requested behavior for Persian study narration. A fully Arabic quotation that
needs classical or regional Arabic pronunciation is a different narration
language and should not be judged by this Farsi pipeline.

Use `[translation.protect]` for technical terms, code identifiers, products,
and names that must **not** be translated. It overrides
`[translation.glossary]` and keeps the configured source spelling in the
reviewable Persian text. NLLB protects those phrases with copied placeholders;
Qwen receives the same exact-source requirement. Add the exact spelling from
the book, including capitalisation where it matters:

```toml
[translation.protect]
"Hennessy" = true
"Patterson" = true
"PostgreSQL" = true
"B-tree" = true
```

Use `[translation.glossary]` only for a term you deliberately want translated
to a fixed Persian form. Do not put English pronunciation replacements in
either table: those apply only to the original Kokoro English reader and are
intentionally disabled for Persian output. Keeping Latin source spelling does
not make a single Persian Piper voice pronounce arbitrary English names
naturally; high-quality code switching would require separate English voice
segments or a convincingly bilingual Persian model.

## Memory and speed

Translation and narration are sequential. NLLB/Qwen is released before Piper
loads, so the translation and Persian TTS models do not compete for unified
memory. Translation is normally slower than Kokoro TTS; reduce
`translation.chunk_chars` for a smaller working set, or select the `low`
resource profile when the Mac is busy. The JSON sidecar reports token counts,
duration, backend, and model; MLX-memory fields apply only to the Qwen backend.

Long-running commands show progress on the terminal's standard error stream,
leaving the final JSON result on standard output. Extraction reports pages per
second; translation reports segments per second; narration and benchmarks show
generated-audio seconds per wall-clock second as `x realtime`.
