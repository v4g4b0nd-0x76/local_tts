# Local two-voice study podcasts

`--podcast` turns a selected page range or chapter into a short, source-grounded
conversation. It uses the same bounded context planner as `--summarize`: the
selected pages are primary evidence; the immediately previous outline chapter
and explicit `Chapter N` references are optional context only.

```bash
uv sync --extra summarize
uv run local-tts book.pdf --config config/natural-explanatory-reader.toml \
  --pages 120-160 --podcast
```

For each independent chapter/range output, the command writes:

- `label-podcast.md`: readable speaker-labelled transcript.
- `label-podcast.json`: exact included pages, local-model timing/memory/token
  measurements, selected voices, and the structured turns.
- `label-podcast.m4b` (or the selected audio format): one combined audio file
  with alternating voices and chunk-level `--resume` support.

## American-English sample pair

The supplied `natural-explanatory-reader.toml` uses the verified Kokoro MLX
voices `af_bella` for the host and `am_michael` for the explainer. Their `af_`
and `am_` prefixes select American-English voice styles. The normal book reader
voice is separate, so retaining the sample's calm `bf_emma` narration does not
change the podcast accents.

Override one run without changing the config:

```bash
uv run local-tts book.pdf --pages 120-160 --podcast \
  --podcast-host-voice af_bella \
  --podcast-explainer-voice am_michael \
  --podcast-max-turns 8 --podcast-max-tokens 640
```

`[podcast]` supports `host_voice`, `explainer_voice`, `max_output_tokens`,
`max_turns`, and `turn_pause_ms`. The generator accepts only a strict,
alternating JSON script (`host`, then `explainer`); malformed model output stops
instead of quietly narrating an unverified or mixed-up transcript.

## Local cost and memory behavior

The local MLX language model writes the dialogue first and is fully released
before Kokoro loads. The model phases therefore do not overlap in unified
memory. Cost grows roughly with the requested source context and generated
dialogue tokens; a larger turn limit mainly permits a longer script. The
podcast JSON records the actual token counts, wall time, and MLX active/peak
memory for each run. Kokoro then synthesizes each turn serially, with only a
bounded PCM checkpoint stream retained on disk.
