import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from utils.retriever import (
    ContextRetriever,
    EmbeddingEncoder,
    IndexStorageManager,
    VectorIndexer,
)


class DummyModel:
    def encode(self, texts):
        vectors = []
        for t in texts:
            v = np.zeros(3, dtype="float32")
            if "cat" in t:
                v[0] = 1.0
            if "dog" in t:
                v[1] = 1.0
            if "bird" in t:
                v[2] = 1.0
            vectors.append(v)
        return np.vstack(vectors)


def build_retriever():
    model = DummyModel()
    encoder = EmbeddingEncoder(model=model)
    indexer = VectorIndexer(dimension=3, metric="cosine")
    metadata = [
        {
            "text": "The cat sat on the mat.",
            "doc_id": "d1",
            "context_id": "c1",
            "section": "methods",
        },
        {
            "text": "Dogs are friendly animals.",
            "doc_id": "d2",
            "context_id": "c2",
            "section": "results",
        },
        {
            "text": "Birds can fly high.",
            "doc_id": "d3",
            "context_id": "c3",
            "section": "discussion",
        },
    ]
    vectors = encoder.encode([m["text"] for m in metadata])
    indexer.add(vectors)
    return ContextRetriever(
        encoder=encoder, indexer=indexer, metadata=metadata
    )


def test_retrieve_basic():
    retriever = build_retriever()
    results = retriever.retrieve("cat", top_k=1)
    assert results
    assert results[0].doc_id == "d1"


def test_retrieve_with_filters():
    retriever = build_retriever()
    results = retriever.retrieve(
        "dog", top_k=2, filters={"section": "results"}
    )
    assert results
    assert results[0].doc_id == "d2"
    assert all(r.section == "results" for r in results)


def test_index_storage_roundtrip(tmp_path):
    retriever = build_retriever()
    storage = IndexStorageManager(
        index_path=tmp_path / "faiss.index",
        metadata_path=tmp_path / "meta.json",
    )
    storage.save(retriever.indexer, retriever.metadata)
    new_retriever = ContextRetriever.from_storage(retriever.encoder, storage)
    results = new_retriever.retrieve("bird", top_k=1)
    assert results[0].doc_id == "d3"
