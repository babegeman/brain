"""Structure-aware recursive text chunker.

Splits on natural boundaries (paragraphs, newlines, sentences, then characters)
and records char offsets for citation.
"""

from __future__ import annotations

from brain.models import Chunk


# Separators ordered by preference (coarsest to finest)
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_document(
    doc_id: str,
    text: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks with char-offset tracking.

    Args:
        doc_id: Parent document ID.
        text: Full document text.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        metadata: Extra metadata to attach to each chunk.

    Returns:
        List of Chunk objects in document order.
    """
    if not text.strip():
        return []

    raw_chunks = _split_recursive(text, chunk_size, _SEPARATORS)
    chunks: list[Chunk] = []
    offset = 0

    for i, chunk_text in enumerate(raw_chunks):
        # Find the actual start position in the original text
        start = text.find(chunk_text, offset)
        if start == -1:
            start = offset  # fallback
        end = start + len(chunk_text)

        chunks.append(
            Chunk(
                doc_id=doc_id,
                text=chunk_text,
                index=i,
                start_char=start,
                end_char=end,
                metadata=metadata or {},
            )
        )
        # Advance offset but allow overlap
        offset = max(start + 1, end - chunk_overlap)

    return chunks


def _split_recursive(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    """Recursively split text using the coarsest separator that fits."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else [""]

    if not sep:
        # Character-level split (last resort)
        parts = []
        for i in range(0, len(text), chunk_size):
            part = text[i : i + chunk_size].strip()
            if part:
                parts.append(part)
        return parts

    pieces = text.split(sep)
    result: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = len(piece) + len(sep)

        if current_len + piece_len > chunk_size and current:
            merged = sep.join(current).strip()
            if merged:
                if len(merged) > chunk_size:
                    # Still too big — recurse with finer separator
                    result.extend(_split_recursive(merged, chunk_size, remaining_seps))
                else:
                    result.append(merged)
            current = [piece]
            current_len = len(piece)
        else:
            current.append(piece)
            current_len += piece_len

    # Flush remainder
    if current:
        merged = sep.join(current).strip()
        if merged:
            if len(merged) > chunk_size:
                result.extend(_split_recursive(merged, chunk_size, remaining_seps))
            else:
                result.append(merged)

    return result
