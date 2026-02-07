"""Shared domain models used across the system."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class DocType(str, Enum):
    DOCX = "docx"
    PPTX = "pptx"
    TEXT = "text"
    IMAGE = "image"


class Document(BaseModel):
    """A source document after normalisation, before chunking."""

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str
    doc_type: DocType
    title: str
    text: str
    metadata: dict = {}
    content_hash: str = ""
    mtime: float = 0.0
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def compute_hash(self) -> str:
        """SHA-256 of the source file content."""
        path = Path(self.source_path)
        if path.exists():
            self.content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.mtime = path.stat().st_mtime
        return self.content_hash


class Chunk(BaseModel):
    """A chunk of a document, ready for embedding."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = {}


class Citation(BaseModel):
    """A citation pointing back to a source chunk."""

    doc_id: str
    chunk_id: str
    source_path: str
    title: str
    chunk_index: int
    snippet: str
    score: float = 0.0
