"""Tests for brain.stores.vectorstore."""

from __future__ import annotations

import pytest

from brain.models import Chunk
from brain.stores.vectorstore import VectorStore


@pytest.fixture
def vstore():
    store = VectorStore(in_memory=True)
    yield store
    store.close()


_UUID1 = "00000000-0000-0000-0000-000000000001"
_UUID2 = "00000000-0000-0000-0000-000000000002"
_UUID3 = "00000000-0000-0000-0000-000000000003"


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id=_UUID1, doc_id="doc-1", text="Alpha", index=0, start_char=0, end_char=5),
        Chunk(chunk_id=_UUID2, doc_id="doc-1", text="Bravo", index=1, start_char=6, end_char=11),
        Chunk(chunk_id=_UUID3, doc_id="doc-2", text="Charlie", index=0, start_char=0, end_char=7),
    ]


def _fake_embeddings(n: int, dim: int = 1024) -> list[list[float]]:
    """Generate simple distinct embeddings."""
    return [[float(i) / dim] * dim for i in range(n)]


def test_upsert_and_count(vstore, sample_chunks):
    embeddings = _fake_embeddings(3)
    vstore.upsert_chunks(sample_chunks, embeddings)
    assert vstore.count() == 3


def test_search_returns_results(vstore, sample_chunks):
    embeddings = _fake_embeddings(3)
    vstore.upsert_chunks(sample_chunks, embeddings)

    query_emb = [0.0 / 1024] * 1024  # Closest to chunk c-1
    results = vstore.search(query_emb, top_k=2)
    assert len(results) == 2
    assert all("chunk_id" in r for r in results)
    assert all("doc_id" in r for r in results)
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)


def test_search_empty_store(vstore):
    query_emb = [0.0] * 1024
    results = vstore.search(query_emb, top_k=5)
    assert results == []


def test_delete_by_doc_id(vstore, sample_chunks):
    embeddings = _fake_embeddings(3)
    vstore.upsert_chunks(sample_chunks, embeddings)
    assert vstore.count() == 3

    vstore.delete_by_doc_id("doc-1")
    assert vstore.count() == 1  # Only doc-2's chunk remains
