"""Conservative cleanup and local spoken summaries for technical PDF text."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from .models import CleanupOptions

_URL = re.compile(r"(?:\b(?:visit|see|available\s+at)\s+)?(?:https?://\S+|www\.\S+)", re.IGNORECASE)
_CITATION = re.compile(r"\[(?:\d+(?:\s*[,;-]\s*\d+)*)\]|\([^)]*\b(?:19|20)\d{2}[a-z]?[^)]*\)")
_FOOTNOTE = re.compile(r"^\s*(?:\d+|[*†‡])\s+.+$")
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d+\s*$", re.IGNORECASE)
_SCHEMA = re.compile(r"\b(?:CREATE\s+TABLE|ALTER\s+TABLE|CREATE\s+(?:UNIQUE\s+)?INDEX|PRIMARY\s+KEY|FOREIGN\s+KEY)\b", re.IGNORECASE)
_CODE = re.compile(r"(?:^\s{4,}|[{};]|\b(?:def|class|function|func|SELECT|INSERT|UPDATE|DELETE|FROM|return|import)\b)", re.IGNORECASE)
_TABLE = re.compile(r"^\s*\S+(?:\s*\|\s*|\t|\s{3,})\S+")
_TABLE_NAME = re.compile(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"`\[]?([A-Za-z_][\w$]*)", re.IGNORECASE)
_INDEX_NAME = re.compile(r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\s+([A-Za-z_][\w$]*)", re.IGNORECASE)
_FUNCTION = re.compile(r"\b(?:def|function|func)\s+([A-Za-z_][\w$]*)", re.IGNORECASE)
_CLASS = re.compile(r"\bclass\s+([A-Za-z_][\w$]*)", re.IGNORECASE)
_FROM = re.compile(r"\bFROM\s+([A-Za-z_][\w$]*)", re.IGNORECASE)


def repeated_margin_lines(page_texts: Iterable[str], threshold: int = 2) -> set[str]:
    """Find likely recurring headers/footers without deleting ordinary prose."""
    candidates: Counter[str] = Counter()
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidates.update(set(lines[:2]) | set(lines[-2:]))
    return {line for line, count in candidates.items() if count >= threshold and not _PAGE_NUMBER.match(line)}


def clean_page(text: str, options: CleanupOptions, margins: set[str] | None = None) -> str:
    """Clean prose and apply explicit skip/read/explain technical-block policy.

    Explanations are deterministic local summaries, not a hidden LLM pass.
    They preserve the surrounding study flow while signalling that a code,
    schema, or table appeared in the source.
    """
    margins = margins or set()
    lines = _normalise_lines(text)
    kept: list[str] = []
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        if not line or (options.headers_footers and (line in margins or _PAGE_NUMBER.match(line))):
            index += 1
            continue
        if options.footnotes == "skip" and _FOOTNOTE.match(line):
            index += 1
            continue
        kind = _technical_kind(raw_line)
        if kind is None:
            kept.append(line)
            index += 1
            continue

        block, index = _collect_block(lines, index, kind)
        policy = _technical_policy(kind, options)
        if policy == "read":
            kept.append(" ".join(part.strip() for part in block))
        elif policy == "explain":
            kept.append(_explain_block(kind, block))
        # `skip` deliberately emits nothing.

    result = " ".join(kept)
    if options.urls == "skip":
        result = _URL.sub("", result)
    if options.citations == "skip":
        result = _CITATION.sub("", result)
    return re.sub(r"\s+", " ", result).strip()


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


def _collect_block(lines: list[str], start: int, kind: str) -> tuple[list[str], int]:
    block: list[str] = []
    index = start
    while index < len(lines) and _technical_kind(lines[index]) == kind:
        block.append(lines[index])
        index += 1
    return block, index


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
