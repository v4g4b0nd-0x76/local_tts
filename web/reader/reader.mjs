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

document.getElementById("documentName").textContent = name;
document.title = `${name} — local-tts`;

const eventBus = new EventBus();
const linkService = new PDFLinkService({ eventBus });
const pdfViewer = new PDFViewer({
  container: viewerContainer,
  viewer: viewerElement,
  eventBus,
  linkService,
  textLayerMode: 2,
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
