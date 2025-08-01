"""Persistent storage for FAISS index and context metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .vector_indexer import VectorIndexer


class IndexStorageManager:
    """Handle saving and loading of vector indices and metadata."""

    def __init__(
        self, index_path: str | Path, metadata_path: str | Path
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

    def save(self, indexer: VectorIndexer, metadata: List[dict]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        indexer.save(self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    def load(self) -> Tuple[VectorIndexer, List[dict]]:
        indexer = VectorIndexer.load(self.index_path)
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = []
        return indexer, metadata
