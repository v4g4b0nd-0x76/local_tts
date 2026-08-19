from pathlib import Path

from pypdf import PdfWriter

from local_tts.pdf import Chapter, PDFDocument, _chapters_from_starts, _unique_outline_starts, chapter_pages, page_range


def test_page_range_is_one_based_and_inclusive() -> None:
    assert page_range("2-4", 10) == [2, 3, 4]


def test_chapter_range_covers_complete_chapters() -> None:
    chapters = [Chapter(1, "One", 1, 3), Chapter(2, "Two", 4, 8)]
    assert chapter_pages(chapters, "1-2") == (list(range(1, 9)), "chapters-01-02")


def test_pdf_document_reuses_its_reader_for_inspection_and_extraction(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)
    with PDFDocument(path) as document:
        reader = document.reader
        page_count, _ = document.inspect()
        assert page_count == 1
        assert document.reader is reader
        assert document.extract_pages([1]) == [(1, "")]


def test_duplicate_outline_destinations_do_not_create_inverted_chapters() -> None:
    starts = _unique_outline_starts([("Front matter", 7), ("Acknowledgments", 7), ("Chapter one", 12)])
    chapters = _chapters_from_starts(starts, 20)
    assert chapters == [Chapter(1, "Front matter", 7, 11), Chapter(2, "Chapter one", 12, 20)]
