"""
Unit tests for src/embedding.py.

--------------------------------------------------------------------------
IMPROVEMENT OVER v1: no tests existed for embedding.py in v1. This closes
that gap. The embedding *model* itself is mocked (a fake object with a
deterministic .encode()) so these tests run offline and fast, without
downloading sentence-transformers weights -- ChromaDB itself is real
(not mocked), since it's fast, local, and worth testing against the
actual library rather than a mock of it.
--------------------------------------------------------------------------

Run with:
    pytest tests/test_embedding.py -v
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding import embed_chunks, build_chroma_store  # noqa: E402


class _FakeEmbeddingModel:
    """Deterministic stand-in for a SentenceTransformer: same text always
    produces the same vector, different texts produce different vectors,
    without needing network access or real model weights."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        vectors = []
        for text in texts:
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            vectors.append(rng.rand(self.dim).astype(np.float32))
        return np.array(vectors)


@pytest.fixture
def sample_chunk_records():
    return [
        {"complaint_id": 1, "product_category": "Credit Card", "chunk_index": 0,
         "total_chunks": 1, "chunk_text": "credit card charged twice"},
        {"complaint_id": 2, "product_category": "Personal Loan", "chunk_index": 0,
         "total_chunks": 1, "chunk_text": "loan denied unfairly"},
    ]


def test_embed_chunks_adds_embedding_key(sample_chunk_records):
    model = _FakeEmbeddingModel()
    result = embed_chunks(model, sample_chunk_records)

    assert all("embedding" in r for r in result)
    assert all(len(r["embedding"]) == 384 for r in result)


def test_embed_chunks_preserves_record_count(sample_chunk_records):
    model = _FakeEmbeddingModel()
    result = embed_chunks(model, sample_chunk_records)
    assert len(result) == len(sample_chunk_records)


def test_embed_chunks_different_text_different_embedding(sample_chunk_records):
    model = _FakeEmbeddingModel()
    result = embed_chunks(model, sample_chunk_records)
    assert result[0]["embedding"] != result[1]["embedding"]


def test_embed_chunks_handles_empty_list():
    model = _FakeEmbeddingModel()
    result = embed_chunks(model, [])
    assert result == []


def test_build_chroma_store_indexes_all_chunks(sample_chunk_records):
    model = _FakeEmbeddingModel()
    records = embed_chunks(model, sample_chunk_records)

    with tempfile.TemporaryDirectory() as tmpdir:
        collection = build_chroma_store(records, persist_dir=tmpdir, collection_name="test_collection")
        assert collection.count() == len(sample_chunk_records)


def test_build_chroma_store_preserves_metadata(sample_chunk_records):
    model = _FakeEmbeddingModel()
    records = embed_chunks(model, sample_chunk_records)

    with tempfile.TemporaryDirectory() as tmpdir:
        collection = build_chroma_store(records, persist_dir=tmpdir, collection_name="test_collection")
        result = collection.get(ids=["1_0"], include=["metadatas", "documents"])

        assert result["metadatas"][0]["complaint_id"] == "1"
        assert result["metadatas"][0]["product_category"] == "Credit Card"
        assert result["documents"][0] == "credit card charged twice"


def test_build_chroma_store_is_idempotent_on_rerun(sample_chunk_records):
    # Re-running against the same persist_dir/collection_name should not
    # error or double the chunk count -- this is what makes it safe to
    # re-run after a preprocessing fix without manual cleanup.
    model = _FakeEmbeddingModel()
    records = embed_chunks(model, sample_chunk_records)

    with tempfile.TemporaryDirectory() as tmpdir:
        build_chroma_store(records, persist_dir=tmpdir, collection_name="test_collection")
        collection = build_chroma_store(records, persist_dir=tmpdir, collection_name="test_collection")
        assert collection.count() == len(sample_chunk_records)


def test_build_chroma_store_ids_are_complaint_id_and_chunk_index():
    model = _FakeEmbeddingModel()
    records = embed_chunks(model, [
        {"complaint_id": 5, "product_category": "Money Transfer", "chunk_index": 0,
         "total_chunks": 2, "chunk_text": "first chunk"},
        {"complaint_id": 5, "product_category": "Money Transfer", "chunk_index": 1,
         "total_chunks": 2, "chunk_text": "second chunk"},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        collection = build_chroma_store(records, persist_dir=tmpdir, collection_name="test_collection")
        result = collection.get(ids=["5_0", "5_1"])
        assert set(result["ids"]) == {"5_0", "5_1"}