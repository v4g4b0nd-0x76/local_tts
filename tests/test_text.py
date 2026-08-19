from local_tts.models import CleanupOptions
from local_tts.text import chunk_text, clean_page, repeated_margin_lines


def test_cleanup_dehyphenates_and_skips_urls_and_code() -> None:
    text = "Header\nNet-\nworking is useful. Visit https://example.test.\n    def noop():\n42\n"
    cleaned = clean_page(text, CleanupOptions(), {"Header"})
    assert cleaned == "Networking is useful."


def test_chunking_keeps_all_content_with_a_bound() -> None:
    chunks = chunk_text("One sentence. Two sentence. Three sentence.", 30)
    assert " ".join(chunks) == "One sentence. Two sentence. Three sentence."
    assert max(map(len, chunks)) <= 30


def test_recurring_margins_are_detected() -> None:
    assert "Book title" in repeated_margin_lines(["Book title\nA\n1", "Book title\nB\n2"])


def test_schema_explanation_retains_table_name_and_fields() -> None:
    text = "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);\nThe table stores accounts."
    cleaned = clean_page(text, CleanupOptions(schemas="explain"))
    assert cleaned == "Schema definition for the users table, with fields id, email. The table stores accounts."


def test_code_and_table_explanations_are_local_and_concise() -> None:
    text = "def normalize_name(value):\n    return value.strip()\nname | count\nAda | 3\nDone."
    cleaned = clean_page(text, CleanupOptions(code="explain", tables="explain"))
    assert "Code example defining the function normalize_name." in cleaned
    assert "A table is shown with columns name, count." in cleaned
    assert cleaned.endswith("Done.")
