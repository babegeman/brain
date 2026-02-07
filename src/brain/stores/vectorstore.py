"""Qdrant embedded-mode vector store."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from brain.config import get_settings
from brain.models import Chunk


class VectorStore:
    """Qdrant vector store — runs in embedded mode (no server needed)."""

    def __init__(
        self,
        path: str | None = None,
        collection: str | None = None,
        *,
        in_memory: bool = False,
    ):
        cfg = get_settings().qdrant
        self.collection = collection or cfg.collection

        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            store_path = path or cfg.path
            Path(store_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=store_path)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            embed_dim = get_settings().llm.embed_dim
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=embed_dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunk embeddings with metadata in payload."""
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "index": chunk.index,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """Return top_k nearest chunks with scores."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "doc_id": r.payload["doc_id"],
                "text": r.payload["text"],
                "index": r.payload.get("index", 0),
                "score": r.score,
                "metadata": r.payload.get("metadata", {}),
            }
            for r in results.points
        ]

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all vectors for a given document."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

    def count(self) -> int:
        """Return the total number of vectors in the collection."""
        info = self.client.get_collection(self.collection)
        return info.points_count

    def close(self) -> None:
        self.client.close()
