"""Tests for brain.stores.docstore."""

from __future__ import annotations

import pytest

from brain.models import Chunk, DocType, Document
from brain.stores.docstore import DocStore


@pytest.fixture
def store(tmp_path):
    db = DocStore(str(tmp_path / "test.db"))
    yield db
    db.close()


@pytest.fixture
def sample_doc():
    return Document(
        doc_id="doc-1",
        source_path="/tmp/test.txt",
        doc_type=DocType.TEXT,
        title="Test Doc",
        text="Hello world",
        content_hash="abc123",
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="c-1", doc_id="doc-1", text="Hello", index=0, start_char=0, end_char=5),
        Chunk(chunk_id="c-2", doc_id="doc-1", text="world", index=1, start_char=6, end_char=11),
    ]


def test_upsert_and_get_document(store, sample_doc):
    store.upsert_document(sample_doc)
    retrieved = store.get_document("doc-1")
    assert retrieved is not None
    assert retrieved.title == "Test Doc"
    assert retrieved.source_path == "/tmp/test.txt"


def test_get_document_by_path(store, sample_doc):
    store.upsert_document(sample_doc)
    retrieved = store.get_document_by_path("/tmp/test.txt")
    assert retrieved is not None
    assert retrieved.doc_id == "doc-1"


def test_get_missing_document(store):
    assert store.get_document("nonexistent") is None


def test_list_documents(store, sample_doc):
    store.upsert_document(sample_doc)
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].doc_id == "doc-1"


def test_is_unchanged(store, sample_doc):
    store.upsert_document(sample_doc)
    assert store.is_unchanged("/tmp/test.txt", "abc123") is True
    assert store.is_unchanged("/tmp/test.txt", "different") is False
    assert store.is_unchanged("/tmp/other.txt", "abc123") is False


def test_upsert_and_get_chunks(store, sample_doc, sample_chunks):
    store.upsert_document(sample_doc)
    store.upsert_chunks(sample_chunks)

    chunk = store.get_chunk("c-1")
    assert chunk is not None
    assert chunk.text == "Hello"

    all_chunks = store.get_chunks_for_doc("doc-1")
    assert len(all_chunks) == 2
    assert all_chunks[0].index == 0
    assert all_chunks[1].index == 1


def test_delete_doc_chunks(store, sample_doc, sample_chunks):
    store.upsert_document(sample_doc)
    store.upsert_chunks(sample_chunks)

    store.delete_doc_chunks("doc-1")
    assert store.get_chunks_for_doc("doc-1") == []


def test_upsert_document_replaces(store, sample_doc):
    store.upsert_document(sample_doc)
    sample_doc.title = "Updated Title"
    store.upsert_document(sample_doc)

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].title == "Updated Title"
