"""Conservative cleanup and local spoken summaries for technical PDF text."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Mapping

from .models import CleanupOptions

_URL = re.compile(r"(?:\b(?:visit|see|available\s+at)\s+)?(?:https?://\S+|www\.\S+)", re.IGNORECASE)
_CITATION = re.compile(r"\[(?:\d+(?:\s*[,;-]\s*\d+)*)\]|\([^)]*\b(?:19|20)\d{2}[a-z]?[^)]*\)")
_FOOTNOTE = re.compile(r"^\s*(?:\d+|[*†‡])\s+.+$")
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d+\s*$", re.IGNORECASE)
_SCHEMA = re.compile(
    r"(?:\b(?:CREATE\s+TABLE|ALTER\s+TABLE|CREATE\s+(?:UNIQUE\s+)?INDEX)\b|^\s*(?:PRIMARY|FOREIGN)\s+KEY\s*\()",
    re.IGNORECASE,
)
_CODE = re.compile(
    r"(?:^\s{4,}|[{}]|\b(?:char|short|int|long|float|double|void|size_t|u?int\d*_t|struct\s+\w+|enum\s+\w+)\s*\*?\s*\w+(?:\s*\[[^\]]+\])?\s*;|(?:^|\s)//|\w+\s*\([^)]*\)\s*[;{]|\breturn\s*\w*\s*;|(?:^|\s)(?:\$|prompt>)\s*|^\s*(?:def\s+\w+|class\s+\w+|function\s+\w+\s*\(|func\s+\w+|import\s+\w+|SELECT\b|INSERT\b|UPDATE\b|DELETE\b))",
    re.IGNORECASE,
)
_TABLE = re.compile(r"^\s*\S+(?:\s*\|\s*|\t|\s{3,})\S+")
_TABLE_NAME = re.compile(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"`\[]?([A-Za-z_][\w$]*)", re.IGNORECASE)
_INDEX_NAME = re.compile(r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\s+([A-Za-z_][\w$]*)", re.IGNORECASE)
_FUNCTION = re.compile(r"\b(?:def|function|func)\s+([A-Za-z_][\w$]*)", re.IGNORECASE)
_CLASS = re.compile(r"\bclass\s+([A-Za-z_][\w$]*)", re.IGNORECASE)
_FROM = re.compile(r"\bFROM\s+([A-Za-z_][\w$]*)", re.IGNORECASE)
_C_DATA_STRUCTURE = re.compile(r"\b(?:struct|enum)\s+([A-Za-z_][\w$]*)", re.IGNORECASE)


def repeated_margin_lines(page_texts: Iterable[str], threshold: int = 2) -> set[str]:
    """Find likely recurring headers/footers without deleting ordinary prose."""
    candidates: Counter[str] = Counter()
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidates.update(set(lines[:2]) | set(lines[-2:]))
    return {line for line, count in candidates.items() if count >= threshold and not _PAGE_NUMBER.match(line)}


def clean_page(
    text: str,
    options: CleanupOptions,
    margins: set[str] | None = None,
    layout_kinds: Mapping[str, str] | None = None,
) -> str:
    """Clean prose and apply explicit skip/read/explain technical-block policy.

    Explanations are deterministic local summaries, not a hidden LLM pass.
    They preserve the surrounding study flow while signalling that a code,
    schema, or table appeared in the source.
    """
    margins = margins or set()
    lines = _normalise_lines(text)
    kept: list[str] = []
    explanations: set[str] = set()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        if not line or (options.headers_footers and (line in margins or _PAGE_NUMBER.match(line))):
            index += 1
            continue
        kind = _line_kind(raw_line, layout_kinds)
        if options.footnotes == "skip" and kind is None and _FOOTNOTE.match(line):
            index += 1
            continue
        if kind is None:
            kept.append(line)
            index += 1
            continue

        block, index = _collect_block(lines, index, kind, layout_kinds)
        policy = _technical_policy(kind, options)
        if policy == "read":
            kept.append(" ".join(part.strip() for part in block))
        elif policy == "explain":
            explanation = _explain_block(kind, block)
            # A PDF may split one visual code listing into fragments (for
            # example source, terminal output, then a second command). The
            # same generic notice should not be spoken repeatedly on a page.
            if explanation not in explanations:
                kept.append(explanation)
                explanations.add(explanation)
        # `skip` deliberately emits nothing.

    result = " ".join(kept)
    if options.urls == "skip":
        result = _URL.sub("", result)
    if options.citations == "skip":
        result = _CITATION.sub("", result)
    return re.sub(r"\s+", " ", result).strip()


def apply_pronunciations(text: str, replacements: tuple[tuple[str, str], ...]) -> str:
    """Apply user-owned spoken-spelling rewrites immediately before TTS.

    PDF extraction can separate a word at an awkward location. The config is
    intentionally a spelling-to-spelling dictionary rather than a hidden
    phoneme API: the rendered text stays inspectable and portable between TTS
    backends.
    """
    for source, spoken in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        words = source.strip()
        if not words:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in words.split()) + r"(?!\w)"
        text = re.sub(pattern, spoken, text, flags=re.IGNORECASE)
    return text


def _normalise_lines(text: str) -> list[str]:
    source_lines = text.replace("\u00ad", "").splitlines()
    lines: list[str] = []
    # Rejoin an obvious PDF line-wrap hyphen, but retain block line boundaries
    # until after code/schema/table detection.
    for raw_line in source_lines:
        if lines and re.search(r"\w-$", lines[-1].rstrip()) and re.match(r"\s*\w", raw_line):
            lines[-1] = lines[-1].rstrip()[:-1] + raw_line.lstrip()
        else:
            lines.append(raw_line)
    return lines


def _technical_kind(line: str) -> str | None:
    if _SCHEMA.search(line):
        return "schema"
    if _CODE.search(line):
        return "code"
    if _TABLE.search(line):
        return "table"
    return None


def _collect_block(
    lines: list[str], start: int, kind: str, layout_kinds: Mapping[str, str] | None
) -> tuple[list[str], int]:
    block: list[str] = []
    index = start
    while index < len(lines) and _line_kind(lines[index], layout_kinds) == kind:
        block.append(lines[index])
        index += 1
    return block, index


def _line_kind(line: str, layout_kinds: Mapping[str, str] | None) -> str | None:
    normalised = " ".join(line.split())
    layout_kind = (layout_kinds or {}).get(normalised)
    text_kind = _technical_kind(line)
    # Layout is better at tables, but normal text can sometimes retain source
    # spacing that layout extraction compresses. Never let a weak table match
    # override a concrete code or schema signature.
    if layout_kind == "table" and text_kind in {"code", "schema"}:
        return text_kind
    return layout_kind or text_kind


def _technical_policy(kind: str, options: CleanupOptions) -> str:
    return {"code": options.code, "schema": options.schemas, "table": options.tables}[kind]


def _explain_block(kind: str, block: list[str]) -> str:
    joined = " ".join(part.strip() for part in block)
    if kind == "schema":
        table = _TABLE_NAME.search(joined)
        if table:
            fields = _schema_fields(joined)
            if fields:
                return f"Schema definition for the {table.group(1)} table, with fields {', '.join(fields[:8])}."
            return f"Schema definition for the {table.group(1)} table."
        index = _INDEX_NAME.search(joined)
        if index:
            return f"Database index definition named {index.group(1)}."
        return "Database schema definition omitted."
    if kind == "table":
        headers = _table_headers(block)
        return f"A table is shown with columns {', '.join(headers)}." if headers else "A table is shown here."
    if "prompt>" in joined or re.search(r"(?:^|\s)\$\s", joined):
        return "A shell session demonstrates compiling or running the example program."
    if "printf" in joined and "while" in joined:
        return "C code example that repeatedly prints a value in a loop."
    data_structure = _C_DATA_STRUCTURE.search(joined)
    if data_structure:
        return f"C data-structure definition for {data_structure.group(1)}."
    function = _FUNCTION.search(joined)
    if function:
        return f"Code example defining the function {function.group(1)}."
    cls = _CLASS.search(joined)
    if cls:
        return f"Code example defining the class {cls.group(1)}."
    source = _FROM.search(joined)
    if source:
        return f"SQL query reading from {source.group(1)}."
    if re.search(r"\bimport\b", joined, re.IGNORECASE):
        return "Code example importing dependencies."
    return "A code example is omitted; it illustrates the surrounding concept."


def _schema_fields(text: str) -> list[str]:
    match = re.search(r"\((.*)\)", text)
    if not match:
        return []
    fields: list[str] = []
    for item in match.group(1).split(","):
        name = item.strip().split(maxsplit=1)[0].strip('"`[]') if item.strip() else ""
        if name and name.upper() not in {"PRIMARY", "FOREIGN", "CONSTRAINT", "UNIQUE", "CHECK"}:
            fields.append(name)
    return fields


def _table_headers(block: list[str]) -> list[str]:
    first = block[0].strip()
    parts = first.split("|") if "|" in first else re.split(r"\t|\s{3,}", first)
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) == 1:
        # Plain extraction often collapses the spaces from a layout-recognised
        # header like `Pass(A)  Pass(B)  Who Runs?`. Keep the labels legible.
        parts = re.split(r"(?<=\))\s+(?=[A-Z])", first)
    return [part.strip() for part in parts if part.strip()][:8]


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split on sentences where possible, with a hard character bound."""
    if max_chars < 20:
        raise ValueError("chunk size must be at least 20 characters")
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        while len(sentence) > max_chars:
            space = sentence.rfind(" ", 0, max_chars)
            split_at = space if space > max_chars // 2 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
