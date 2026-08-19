# Technical-reading policy and narration style

Use [natural-explanatory-reader.toml](../config/natural-explanatory-reader.toml)
as the slower, smoother study-book profile:

```bash
uv run local-tts book.pdf --config config/natural-explanatory-reader.toml --chapters 3-5
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
Layout-preserving extraction improves detection of indented source blocks,
schemas, and aligned tables; see [PDF layout and metadata](pdf-layout.md) for
the conservative matching rules.

## Narration controls

Kokoro MLX has deterministic voice-style files and does **not** expose a
random seed. The controls that affect delivery are:

- `reader.voice`: voice/style, including its language/accent prefix.
- `reader.speed`: rate multiplier; lower values sound less rushed. The sample
  can be set to `0.92` for a smoother British-English technical narration.
- `reader.chunk_pause_ms`: a small silence at durable chunk boundaries. The
  sample uses 180 ms; it adds a little audio length without more AI or hardware
  resource use.
- `reader.sample_rate`: use 24 kHz native output. Selecting 48 kHz upsamples
  the waveform but does not improve model detail and costs extra work.

`[pronunciation]` is a per-config spelling rewrite applied immediately before
Kokoro. It repairs extraction such as `prob lem` or `prob-lem` into the normal
word `problem`; the supplied British-English G2P already has that ordinary word
in its lexicon. Add only words you have heard pronounced incorrectly. The text
is not sent anywhere and the mapping becomes part of the resume hash.

When `explain` sees the same generic code or shell notice more than once on a
page, it speaks that exact notice once. Distinct explanatory statements remain.

Override any setting for one run without editing the file:

```bash
uv run local-tts book.pdf --config config/study-reader.toml \
  --voice af_bella --speed 0.94 --code skip --schemas explain
```
