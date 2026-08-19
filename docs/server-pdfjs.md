# Local streaming server and PDF.js reader

`local-tts` can open a selected PDF in a local browser reader and stream speech
for selections, the current page, or sequential reading mode. It contains no
authentication because the default bind address is `127.0.0.1`; do not use
`--allow-network` unless you deliberately want to expose your reader to another
machine.

## Open a PDF in the local reader

```bash
cd ~/projects/local_tts
uv sync --extra server
uv run local-tts setup-viewer       # one-time PDF.js download into web/node_modules
uv run local-tts read ~/Books/book.pdf \
  --config config/natural-explanatory-reader.toml --port 8765
```

`read` starts the loopback service, serves the selected PDF and the locally
installed PDF.js component viewer, and opens it in the default browser. The
browser URL contains only a local reader token, not the PDF's filesystem path.
Use `--no-open` when starting it from another script. `setup-viewer` is the
only command that needs npm/network access; all later PDF viewing and speech is
local.

The reader toolbar has **Read selection**, **Read page**, **Reading mode**, and
**Stop reading**. The page text layer highlights the currently spoken chunk. A
PDF-load progress bar is shown while the document opens. When a reader action
is requested, a second indeterminate “Preparing local speech” bar appears; it
changes to a chunk counter once synthesis begins, then to “Reading” while
queued local audio plays.

The packaged reader uses the Kuro Nezumi theme by default: soot-black browser chrome,
warm paper-colored document text, and signal red only for active controls,
progress, and spoken-text highlighting. PDF.js pages are rendered locally in a
desaturated dark-paper treatment; this intentionally also mutes the colors in
figures and photographs so the whole reading surface stays consistent.

## Zoom quality and appearance

The reader provides 50–600% zoom plus page-width and page-fit controls. It
allows PDF.js to draw up to 128 million pixels per visible page (four times its
normal desktop ceiling) and keeps PDF.js's high-detail partial renderer on.
This makes vector text and diagrams stay sharp at high zoom without allowing an
unbounded canvas allocation. A scanned or otherwise low-resolution PDF image
cannot gain detail that was not present in the source.

Use the standard light PDF.js appearance instead of Kuro Nezumi:

```bash
uv run local-tts read ~/Books/book.pdf --theme default
```

For a local custom stylesheet, start from
[the example](../config/pdf-reader-custom.css.example):

```bash
cp config/pdf-reader-custom.css.example ~/Documents/my-pdf-reader.css
uv run local-tts read ~/Books/book.pdf --theme custom \
  --theme-css ~/Documents/my-pdf-reader.css
```

The custom stylesheet is exposed only to the loopback reader as
`/reader/custom-theme.css`; the URL never reveals its filesystem path. You can
also put the setting in a reader TOML file:

```toml
[viewer]
theme = "custom"
custom_css = "/Users/you/Documents/my-pdf-reader.css"
```

The server loads one Kokoro instance and serializes synthesis. This is
intentional: competing Apple-GPU requests reduce throughput and can cause
unpredictable unified-memory pressure. The supplied TOML controls its voice,
speed, chunk pause, pronunciation mappings, and resource profile. For example,
use `--voice af_bella` for an American-English reader without editing the
configuration.

## API

`GET /healthz` reports the active backend and stream defaults.

`POST /v1/speech/stream` returns `application/x-ndjson`. It first emits a
`start` record with the planned chunk total, then each synthesized chunk as
base64 mono 16-bit PCM with exact source-text `start`/`end` character offsets,
then a `done` record. This is the interactive-reader endpoint.

```bash
curl --no-buffer http://127.0.0.1:8765/v1/speech/stream \
  -H 'content-type: application/json' \
  -d '{"text":"Read this locally with the configured voice.","voice":"af_bella"}'
```

`POST /v1/speech/wav` is a progressive `audio/wav` response for clients that
can consume an open-ended WAV stream. It has no text-offset metadata; use the
NDJSON endpoint for PDF highlighting. Both endpoints are bounded to 50,000
characters per request by default; set `--max-request-chars` only when needed.

## Embed the bridge in another PDF.js app

[pdfjs-local-tts.js](../web/pdfjs-local-tts.js) is also a dependency-free
bridge for another local PDF.js component viewer. Load it after PDF.js and
give it the viewer instance:

```html
<script src="pdfjs-local-tts.js"></script>
<script>
  const localTtsReader = LocalTTSPDFJS.install({
    baseUrl: "http://127.0.0.1:8765",
    pdfViewer,
    pdfDocument,
  });
  localTtsReader.installControls(document.querySelector("#your-toolbar"));
</script>
```

The bridge uses `AudioContext`, not a remote player or cloud service. It
decodes every PCM event locally and schedules it immediately, so a response
does not wait for a full page or chapter. If PDF.js has not rendered a text
layer yet, the bridge falls back to PDF.js extraction and still reads the page,
but cannot show span-level highlighting until that layer is available.

The packaged reader is already on the same loopback origin. For an external
local PDF.js app, `file://` sends the browser origin `null`, and normal local
dev servers use `localhost` or `127.0.0.1`; those origins are enabled by
default. Use `--allow-origin https://your-local-viewer.example` only for a
specific additional viewer origin.
