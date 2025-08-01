import os
import sys
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from utils.retriever import (  # noqa: E402
    ContextRetriever,
    EmbeddingEncoder,
    IndexStorageManager,
    MemoryBuilder,
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


def test_memory_builder_roundtrip(tmp_path):
    contexts = [
        {
            "text": "The cat sat on the mat.",
            "doc_id": "d1",
            "context_id": "c1",
        },
        {
            "text": "Dogs are friendly animals.",
            "doc_id": "d2",
            "context_id": "c2",
        },
    ]
    model = DummyModel()
    encoder = EmbeddingEncoder(model=model)
    builder = MemoryBuilder(encoder=encoder)
    indexer, metadata = builder.build(contexts)
    storage = IndexStorageManager(
        index_path=tmp_path / "faiss.index",
        metadata_path=tmp_path / "meta.json",
    )
    storage.save(indexer, metadata)
    retriever = ContextRetriever.from_storage(encoder, storage)
    results = retriever.retrieve("cat", top_k=1)
    assert results[0].doc_id == "d1"
