/*
 * Drop-in bridge for the official local PDF.js viewer.
 *
 * Load this after viewer.js, then run:
 *   const reader = LocalTTSPDFJS.install();
 *   reader.installControls();
 */
(function (global) {
  "use strict";

  const ACTIVE_CLASS = "local-tts-active";

  class LocalTTSPDFJSReader {
    constructor({ baseUrl = "http://127.0.0.1:8765", pdfViewer, pdfDocument, progressElement, progressLabel } = {}) {
      const app = global.PDFViewerApplication;
      this.viewer = pdfViewer || app?.pdfViewer;
      this.document = pdfDocument || app?.pdfDocument;
      if (!this.viewer) {
        throw new Error("PDF.js viewer is unavailable; load this after viewer.js has initialized.");
      }
      this.baseUrl = baseUrl.replace(/\/$/, "");
      this.audioContext = null;
      this.abortController = null;
      this.sources = new Set();
      this.highlighted = new Set();
      this.timers = new Set();
      this.nextStart = 0;
      this.run = 0;
      this.reading = false;
      this.progressElement = progressElement || null;
      this.progressLabel = progressLabel || null;
      this._installStyle();
    }

    async readSelection(options = {}) {
      const selection = global.getSelection();
      if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
        throw new Error("Select text in the PDF before choosing Read selection.");
      }
      const range = selection.getRangeAt(0);
      const layer = closestTextLayer(range.commonAncestorContainer);
      if (layer && layer.contains(range.startContainer) && layer.contains(range.endContainer)) {
        const map = textMap(layer);
        const start = rangeOffset(layer, range.startContainer, range.startOffset);
        const end = rangeOffset(layer, range.endContainer, range.endOffset);
        if (start !== null && end !== null && end > start) {
          return this.speakText(map.text.slice(start, end), { ...options, map, sourceOffset: start });
        }
      }
      // A multi-page selection is still readable, but PDF.js does not expose a
      // reliable single text-layer map for it, so it retains the native selection.
      return this.speakText(selection.toString(), options);
    }

    async readCurrentPage(options = {}) {
      this.stop();
      return this._readPage(this.viewer.currentPageNumber, this.run, options);
    }

    async startReading({ lastPage = this.viewer.pagesCount, ...options } = {}) {
      this.stop();
      const run = this.run;
      this.reading = true;
      const firstPage = this.viewer.currentPageNumber;
      for (let page = firstPage; page <= lastPage && run === this.run; page += 1) {
        this.viewer.currentPageNumber = page;
        await nextPaint();
        await this._readPage(page, run, options);
      }
      if (run === this.run) this.reading = false;
    }

    async speakText(text, options = {}) {
      this.stop();
      return this._streamText(text, this.run, options);
    }

    stop() {
      this.run += 1;
      this.reading = false;
      this.abortController?.abort();
      this.abortController = null;
      for (const source of this.sources) {
        try { source.stop(); } catch (_) { /* source already ended */ }
      }
      this.sources.clear();
      for (const timer of this.timers) global.clearTimeout(timer);
      this.timers.clear();
      this.nextStart = 0;
      this._clearHighlight();
      this._hideProgress();
    }

    installControls(toolbar = global.document.querySelector("#toolbarViewerRight")) {
      if (!toolbar) throw new Error("PDF.js toolbar was not found.");
      const add = (label, title, handler) => {
        const button = global.document.createElement("button");
        button.type = "button";
        button.className = "toolbarButton";
        button.textContent = label;
        button.title = title;
        button.addEventListener("click", () => handler().catch((error) => global.alert(error.message)));
        toolbar.append(button);
      };
      add("Read selection", "Read selected text locally", () => this.readSelection());
      add("Read page", "Read the current page locally", () => this.readCurrentPage());
      add("Reading mode", "Read from the current page onward locally", () => this.startReading());
      add("Stop reading", "Stop local narration", async () => this.stop());
      if (!this.progressElement) {
        const status = global.document.createElement("span");
        status.className = "local-tts-status";
        const progress = global.document.createElement("progress");
        progress.className = "local-tts-progress";
        progress.hidden = true;
        toolbar.append(status, progress);
        this.progressLabel = status;
        this.progressElement = progress;
      }
      return this;
    }

    async _readPage(pageNumber, run, options) {
      if (run !== this.run) return;
      const layer = await this._textLayer(pageNumber);
      if (run !== this.run) return;
      if (layer) {
        const map = textMap(layer);
        if (map.text.trim()) return this._streamText(map.text, run, { ...options, map, sourceOffset: 0 });
      }
      const text = await this._pageText(pageNumber);
      if (text.trim()) return this._streamText(text, run, options);
    }

    async _textLayer(pageNumber) {
      const pageView = this.viewer.getPageView?.(pageNumber - 1);
      let layer = pageView?.textLayer?.div || pageView?.textLayer?.textLayerDiv;
      if (!layer) {
        layer = global.document.querySelector(`.page[data-page-number="${pageNumber}"] .textLayer`);
      }
      if (layer?.textContent?.trim()) return layer;
      // PDF.js renders a text layer lazily after navigation; give it two paints
      // before using the extraction fallback without page highlighting.
      await nextPaint();
      await nextPaint();
      return global.document.querySelector(`.page[data-page-number="${pageNumber}"] .textLayer`);
    }

    async _pageText(pageNumber) {
      const document = this.document || global.PDFViewerApplication?.pdfDocument;
      if (!document) throw new Error("PDF.js document is unavailable.");
      const page = await document.getPage(pageNumber);
      const content = await page.getTextContent();
      return content.items.map((item) => `${item.str}${item.hasEOL ? "\n" : " "}`).join("");
    }

    async _streamText(text, run, { map = null, sourceOffset = 0, voice, speed, chunkChars } = {}) {
      if (!text.trim() || run !== this.run) return;
      this._setProgress("Preparing local speech…");
      if (!this.audioContext) this.audioContext = new global.AudioContext();
      await this.audioContext.resume();
      const controller = new AbortController();
      this.abortController = controller;
      try {
        const response = await fetch(`${this.baseUrl}/v1/speech/stream`, {
          method: "POST",
          signal: controller.signal,
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ text, voice, speed, chunk_chars: chunkChars }),
        });
        if (!response.ok || !response.body) throw new Error(`Local TTS request failed (${response.status}).`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let pending = "";
        let chunksTotal = 0;
        try {
          while (run === this.run) {
            const { done, value } = await reader.read();
            pending += decoder.decode(value || new Uint8Array(), { stream: !done });
            let newline;
            while ((newline = pending.indexOf("\n")) >= 0) {
              const line = pending.slice(0, newline);
              pending = pending.slice(newline + 1);
              if (!line) continue;
              const event = JSON.parse(line);
              if (event.type === "start") {
                chunksTotal = event.chunks_total || 0;
                this._setProgress("Preparing local speech…", 0, chunksTotal);
              }
              if (event.type === "audio") {
                this._scheduleAudio(event, map, sourceOffset, run);
                this._setProgress(`Preparing speech (${event.sequence + 1}/${chunksTotal})…`, event.sequence + 1, chunksTotal);
              }
              if (event.type === "error") throw new Error(event.message || "Local synthesis failed.");
            }
            if (done) break;
          }
        } finally {
          reader.releaseLock();
        }
        if (run === this.run && this.nextStart > this.audioContext.currentTime) {
          this._setProgress("Reading…", chunksTotal, chunksTotal);
          await wait((this.nextStart - this.audioContext.currentTime) * 1000 + 30);
        }
        if (run === this.run) this.abortController = null;
        this._hideProgress();
      } catch (error) {
        if (run === this.run && error.name !== "AbortError") this._setProgress("Local speech request failed");
        throw error;
      }
    }

    _scheduleAudio(event, map, sourceOffset, run) {
      const samples = decodePCM16(event.audio);
      const buffer = this.audioContext.createBuffer(1, samples.length, event.sample_rate);
      buffer.copyToChannel(samples, 0);
      const source = this.audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioContext.destination);
      const when = Math.max(this.nextStart, this.audioContext.currentTime + 0.05);
      this.nextStart = when + buffer.duration;
      this.sources.add(source);
      source.onended = () => this.sources.delete(source);
      source.start(when);
      if (map) {
        const delay = Math.max(0, (when - this.audioContext.currentTime) * 1000);
        const timer = global.setTimeout(() => {
          this.timers.delete(timer);
          if (run === this.run) this._highlight(map, sourceOffset + event.start, sourceOffset + event.end);
        }, delay);
        this.timers.add(timer);
      }
    }

    _highlight(map, start, end) {
      this._clearHighlight();
      for (const item of map.nodes) {
        if (item.end <= start || item.start >= end) continue;
        const element = item.node.parentElement;
        if (element) {
          element.classList.add(ACTIVE_CLASS);
          this.highlighted.add(element);
        }
      }
    }

    _clearHighlight() {
      for (const element of this.highlighted) element.classList.remove(ACTIVE_CLASS);
      this.highlighted.clear();
    }

    _installStyle() {
      if (global.document.getElementById("local-tts-pdfjs-style")) return;
      const style = global.document.createElement("style");
      style.id = "local-tts-pdfjs-style";
      style.textContent = `.textLayer .${ACTIVE_CLASS} { background: rgba(183, 53, 53, .48) !important; box-shadow: 0 0 0 1px rgba(217, 74, 74, .55); border-radius: 2px; } .local-tts-status { margin-left: 8px; font-size: 12px; } .local-tts-progress { width: 96px; margin: 0 4px; vertical-align: middle; }`;
      global.document.head.append(style);
    }

    _setProgress(label, value, total) {
      if (this.progressLabel) this.progressLabel.textContent = label;
      if (!this.progressElement) return;
      this.progressElement.hidden = false;
      if (Number.isFinite(value) && total > 0) {
        this.progressElement.max = total;
        this.progressElement.value = value;
      } else {
        this.progressElement.removeAttribute("value");
      }
    }

    _hideProgress() {
      if (this.progressLabel) this.progressLabel.textContent = "";
      if (this.progressElement) this.progressElement.hidden = true;
    }
  }

  function textMap(layer) {
    const nodes = [];
    const walker = global.document.createTreeWalker(layer, global.NodeFilter.SHOW_TEXT);
    let offset = 0;
    let node;
    while ((node = walker.nextNode())) {
      const value = node.textContent || "";
      nodes.push({ node, start: offset, end: offset + value.length });
      offset += value.length;
    }
    return { layer, nodes, text: nodes.map((item) => item.node.textContent).join("") };
  }

  function closestTextLayer(node) {
    const element = node.nodeType === global.Node.ELEMENT_NODE ? node : node.parentElement;
    return element?.closest?.(".textLayer") || null;
  }

  function rangeOffset(layer, container, offset) {
    try {
      const before = global.document.createRange();
      before.setStart(layer, 0);
      before.setEnd(container, offset);
      return before.toString().length;
    } catch (_) {
      return null;
    }
  }

  function decodePCM16(base64) {
    const binary = global.atob(base64);
    const data = new DataView(new ArrayBuffer(binary.length));
    for (let index = 0; index < binary.length; index += 1) data.setUint8(index, binary.charCodeAt(index));
    const samples = new Float32Array(binary.length / 2);
    for (let index = 0; index < samples.length; index += 1) samples[index] = data.getInt16(index * 2, true) / 32768;
    return samples;
  }

  function nextPaint() {
    return new Promise((resolve) => global.requestAnimationFrame(() => global.requestAnimationFrame(resolve)));
  }

  function wait(milliseconds) {
    return new Promise((resolve) => global.setTimeout(resolve, milliseconds));
  }

  global.LocalTTSPDFJS = {
    LocalTTSPDFJSReader,
    install(options) { return new LocalTTSPDFJSReader(options); },
  };
}(window));
