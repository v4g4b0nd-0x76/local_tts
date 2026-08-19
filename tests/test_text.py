from local_tts.models import CleanupOptions
from local_tts.text import apply_pronunciations, chunk_text, clean_page, repeated_margin_lines


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


def test_collapsed_layout_table_headings_remain_speakable() -> None:
    text = "Pass(A) Pass(B) Pass(C) Who Runs?\n0 0 0 A\n100 0 0 B"
    cleaned = clean_page(text, CleanupOptions(tables="explain"), layout_kinds={
        "Pass(A) Pass(B) Pass(C) Who Runs?": "table",
        "0 0 0 A": "table",
        "100 0 0 B": "table",
    })
    assert cleaned == "A table is shown with columns Pass(A), Pass(B), Pass(C), Who Runs?."


def test_common_c_loop_and_shell_examples_get_useful_local_summaries() -> None:
    text = "while (1) { printf(\"hello\"); }\n\nprompt> gcc -o app app.c"
    cleaned = clean_page(text, CleanupOptions(code="explain"))
    assert "repeatedly prints a value in a loop" in cleaned
    assert "shell session demonstrates compiling or running" in cleaned


def test_numbered_source_code_is_not_mistaken_for_a_footnote() -> None:
    text = "15 while (1) {\n16 printf(\"hello\");\n17 }"
    cleaned = clean_page(text, CleanupOptions(code="explain", footnotes="skip"))
    assert cleaned == "C code example that repeatedly prints a value in a loop."


def test_prose_with_urls_or_the_word_from_is_not_treated_as_code() -> None:
    text = "Read https://example.test/from-guide before learning from the table."
    assert clean_page(text, CleanupOptions()) == "Read before learning from the table."


def test_repeated_code_notice_is_spoken_once_per_page() -> None:
    text = "prompt> gcc -o app app.c\nThe program now runs.\nprompt> ./app"
    cleaned = clean_page(text, CleanupOptions(code="explain"))
    assert cleaned.count("A shell session demonstrates compiling or running the example program.") == 1
    assert "The program now runs." in cleaned


def test_pronunciation_rewrites_repair_split_words_before_tts() -> None:
    assert apply_pronunciations("A prob lem is solvable.", (("prob lem", "problum"),)) == "A problum is solvable."
    assert apply_pronunciations("The Problem is clear.", (("problem", "problum"),)) == "The problum is clear."
