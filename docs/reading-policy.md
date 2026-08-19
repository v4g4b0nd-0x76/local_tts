# Technical-reading policy and narration style

Use [study-reader.toml](../config/study-reader.toml) as the explicit study-book
profile:

```bash
uv run local-tts book.pdf --config config/study-reader.toml --chapters 3-5
```

## Code, schemas, and tables

Each policy is independently set to `skip`, `explain`, or `read`.

- `skip` removes the technical block from speech.
- `explain` emits a short deterministic local description. For example, a
  `CREATE TABLE users` block becomes “Schema definition for the users table”;
  function and class names, SQL query sources, database index names, and table
  headers are retained where extraction makes them clear.
- `read` passes the extracted text through unchanged. It is useful for a short
  query, but punctuation-heavy source code is usually less pleasant than
  `explain`.

No generative model is called for these summaries. Ambiguous code becomes a
plain notice that an example was omitted, rather than a fabricated explanation.

## Narration controls

Kokoro MLX has deterministic voice-style files and does **not** expose a
random seed. The controls that affect delivery are:

- `reader.voice`: voice/style, including its language/accent prefix.
- `reader.speed`: rate multiplier; lower values sound less rushed. The sample
  uses `bf_emma` at `0.92` for smooth British-English technical narration.
- `reader.sample_rate`: use 24 kHz native output. Selecting 48 kHz upsamples
  the waveform but does not improve model detail and costs extra work.

Override any setting for one run without editing the file:

```bash
uv run local-tts book.pdf --config config/study-reader.toml \
  --voice af_bella --speed 0.94 --code skip --schemas explain
```
