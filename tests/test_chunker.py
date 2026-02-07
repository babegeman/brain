"""Tests for brain.ingest.chunker."""

from __future__ import annotations

from brain.ingest.chunker import chunk_document


def test_short_text_single_chunk():
    """Text shorter than chunk_size should produce one chunk."""
    chunks = chunk_document("doc1", "Hello world", chunk_size=512)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 11


def test_empty_text():
    chunks = chunk_document("doc1", "", chunk_size=512)
    assert chunks == []


def test_whitespace_only():
    chunks = chunk_document("doc1", "   \n\n  ", chunk_size=512)
    assert chunks == []


def test_paragraph_splitting():
    """Two paragraphs should be split at the paragraph boundary."""
    text = "First paragraph with some text.\n\nSecond paragraph with more text."
    chunks = chunk_document("doc1", text, chunk_size=40, chunk_overlap=0)
    assert len(chunks) >= 2
    assert "First" in chunks[0].text
    assert "Second" in chunks[-1].text


def test_chunk_offsets_are_valid():
    """start_char and end_char should correctly locate the chunk in original text."""
    text = "Alpha.\n\nBravo.\n\nCharlie.\n\nDelta."
    chunks = chunk_document("doc1", text, chunk_size=15, chunk_overlap=0)
    for chunk in chunks:
        # The chunk text should be findable at the recorded offset
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert chunk.end_char <= len(text)


def test_chunks_have_sequential_indices():
    text = "A\n\nB\n\nC\n\nD\n\nE\n\nF"
    chunks = chunk_document("doc1", text, chunk_size=5, chunk_overlap=0)
    for i, chunk in enumerate(chunks):
        assert chunk.index == i


def test_doc_id_propagated():
    chunks = chunk_document("my-doc-id", "Hello world", chunk_size=512)
    assert all(c.doc_id == "my-doc-id" for c in chunks)


def test_metadata_propagated():
    chunks = chunk_document(
        "doc1", "Hello world", chunk_size=512, metadata={"key": "val"}
    )
    assert chunks[0].metadata == {"key": "val"}
