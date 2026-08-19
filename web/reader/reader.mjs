import * as pdfjsLib from "/assets/pdfjs/build/pdf.mjs";
import { EventBus, PDFLinkService, PDFViewer } from "/assets/pdfjs/web/pdf_viewer.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "/assets/pdfjs/build/pdf.worker.mjs";

const query = new URLSearchParams(globalThis.location.search);
const file = query.get("file") || "/file";
const name = query.get("name") || "Local PDF reader";
const viewerContainer = document.getElementById("viewerContainer");
const viewerElement = document.getElementById("viewer");
const pageNumber = document.getElementById("pageNumber");
const pageCount = document.getElementById("pageCount");
const loading = document.getElementById("pdfLoading");
const loadingLabel = document.getElementById("pdfLoadingLabel");
const loadingProgress = document.getElementById("pdfLoadingProgress");
const themeSelect = document.getElementById("themeSelect");
const zoomSelect = document.getElementById("zoomSelect");
const zoomSteps = [0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5, 6];

document.getElementById("documentName").textContent = name;
document.title = `${name} — local-tts`;

const customThemeAvailable = query.get("custom_theme") === "1";
let customThemeStylesheet;
if (!customThemeAvailable) {
  themeSelect.querySelector('option[value="custom"]').remove();
}
if (customThemeAvailable) {
  customThemeStylesheet = document.createElement("link");
  customThemeStylesheet.id = "customThemeStylesheet";
  customThemeStylesheet.rel = "stylesheet";
  customThemeStylesheet.href = "/reader/custom-theme.css";
  document.head.append(customThemeStylesheet);
}

function setTheme(requested) {
  const theme = ["kuro-nezumi", "default", "custom"].includes(requested) &&
    (requested !== "custom" || customThemeAvailable)
    ? requested
    : "kuro-nezumi";
  document.documentElement.dataset.readerTheme = theme;
  if (customThemeStylesheet) customThemeStylesheet.disabled = theme !== "custom";
  themeSelect.value = theme;
}

setTheme(query.get("theme") || "kuro-nezumi");
themeSelect.addEventListener("change", () => setTheme(themeSelect.value));

const eventBus = new EventBus();
const linkService = new PDFLinkService({ eventBus });
const pdfViewer = new PDFViewer({
  container: viewerContainer,
  viewer: viewerElement,
  eventBus,
  linkService,
  textLayerMode: 2,
  // PDF.js defaults to 32 MP per page. 128 MP keeps vector pages sharp well
  // beyond 300% zoom while one visible high-detail canvas stays bounded.
  maxCanvasPixels: 2 ** 27,
  enableDetailCanvas: true,
  enableOptimizedPartialRendering: true,
});
linkService.setViewer(pdfViewer);

const localReader = LocalTTSPDFJS.install({
  pdfViewer,
  baseUrl: globalThis.location.origin,
  progressElement: document.getElementById("localTtsProgress"),
  progressLabel: document.getElementById("localTtsStatus"),
});
localReader.installControls(document.getElementById("localTtsControls"));

const loadingTask = pdfjsLib.getDocument({ url: file });
loadingTask.onProgress = ({ loaded, total }) => {
  if (total > 0) {
    loadingProgress.max = total;
    loadingProgress.value = loaded;
    loadingLabel.textContent = `Opening local PDF… ${Math.round((loaded / total) * 100)}%`;
  }
};

const pdfDocument = await loadingTask.promise;
pdfViewer.setDocument(pdfDocument);
linkService.setDocument(pdfDocument, null);
localReader.document = pdfDocument;
pageCount.textContent = `of ${pdfDocument.numPages}`;
loading.hidden = true;

eventBus.on("pagechanging", ({ pageNumber: current }) => {
  pageNumber.value = current;
});

function setZoom(value) {
  pdfViewer.currentScaleValue = value === "page-width" || value === "page-fit"
    ? value
    : Number(value);
}

function stepZoom(direction) {
  const current = pdfViewer.currentScale || 1;
  const next = direction > 0
    ? zoomSteps.find((step) => step > current + 0.01) || zoomSteps.at(-1)
    : [...zoomSteps].reverse().find((step) => step < current - 0.01) || zoomSteps[0];
  setZoom(String(next));
}

zoomSelect.addEventListener("change", () => setZoom(zoomSelect.value));
document.getElementById("zoomIn").addEventListener("click", () => stepZoom(1));
document.getElementById("zoomOut").addEventListener("click", () => stepZoom(-1));
eventBus.on("scalechanging", ({ scale, presetValue }) => {
  if (presetValue === "page-width" || presetValue === "page-fit") {
    zoomSelect.value = presetValue;
    return;
  }
  const nearest = zoomSteps.reduce((best, step) =>
    Math.abs(step - scale) < Math.abs(best - scale) ? step : best,
  zoomSteps[0]);
  zoomSelect.value = String(nearest);
});

document.getElementById("previousPage").addEventListener("click", () => {
  pdfViewer.currentPageNumber = Math.max(1, pdfViewer.currentPageNumber - 1);
});
document.getElementById("nextPage").addEventListener("click", () => {
  pdfViewer.currentPageNumber = Math.min(pdfDocument.numPages, pdfViewer.currentPageNumber + 1);
});
pageNumber.addEventListener("change", () => {
  const requested = Number.parseInt(pageNumber.value, 10);
  pdfViewer.currentPageNumber = Number.isFinite(requested)
    ? Math.min(pdfDocument.numPages, Math.max(1, requested))
    : pdfViewer.currentPageNumber;
});
