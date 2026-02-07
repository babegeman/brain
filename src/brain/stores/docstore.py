"""SQLite-backed document and chunk metadata store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from brain.models import Chunk, Document


class DocStore:
    """Stores document metadata and chunk text in SQLite."""

    def __init__(self, db_path: str = "./data/docstore.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id       TEXT PRIMARY KEY,
                source_path  TEXT NOT NULL,
                doc_type     TEXT NOT NULL,
                title        TEXT NOT NULL,
                text         TEXT NOT NULL,
                metadata     TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT NOT NULL DEFAULT '',
                mtime        REAL NOT NULL DEFAULT 0,
                ingested_at  TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_source ON documents(source_path);

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL REFERENCES documents(doc_id),
                text       TEXT NOT NULL,
                idx        INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char   INTEGER NOT NULL,
                metadata   TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(doc_id);
        """)
        self._conn.commit()

    # -- Documents -----------------------------------------------------------

    def upsert_document(self, doc: Document) -> None:
        """Insert or replace a document record."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO documents
                (doc_id, source_path, doc_type, title, text, metadata, content_hash, mtime, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.doc_id,
                doc.source_path,
                doc.doc_type.value,
                doc.title,
                doc.text,
                json.dumps(doc.metadata),
                doc.content_hash,
                doc.mtime,
                doc.ingested_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Document | None:
        """Fetch a document by ID."""
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return self._row_to_doc(row) if row else None

    def get_document_by_path(self, source_path: str) -> Document | None:
        """Fetch a document by source path."""
        row = self._conn.execute(
            "SELECT * FROM documents WHERE source_path = ?", (source_path,)
        ).fetchone()
        return self._row_to_doc(row) if row else None

    def list_documents(self) -> list[Document]:
        """List all documents."""
        rows = self._conn.execute(
            "SELECT * FROM documents ORDER BY ingested_at DESC"
        ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def is_unchanged(self, source_path: str, content_hash: str) -> bool:
        """Check if a file has already been ingested with the same hash."""
        row = self._conn.execute(
            "SELECT content_hash FROM documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        return row is not None and row["content_hash"] == content_hash

    # -- Chunks --------------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or replace chunk records."""
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, text, idx, start_char, end_char, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.chunk_id,
                    c.doc_id,
                    c.text,
                    c.index,
                    c.start_char,
                    c.end_char,
                    json.dumps(c.metadata),
                )
                for c in chunks
            ],
        )
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Fetch a chunk by ID."""
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_for_doc(self, doc_id: str) -> list[Chunk]:
        """Get all chunks for a document, ordered by index."""
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY idx", (doc_id,)
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def delete_doc_chunks(self, doc_id: str) -> None:
        """Delete all chunks for a document (used before re-ingestion)."""
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_doc(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            source_path=row["source_path"],
            doc_type=row["doc_type"],
            title=row["title"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
            content_hash=row["content_hash"],
            mtime=row["mtime"],
            ingested_at=row["ingested_at"],
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            index=row["idx"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            metadata=json.loads(row["metadata"]),
        )
