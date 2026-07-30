"""
Integration test for src/build_sample_index.py -- runs the full
sample -> chunk -> embed -> index pipeline end to end against a small
synthetic dataset. Only the embedding model is mocked (no network
access needed to download real sentence-transformers weights); sampling,
chunking, and ChromaDB indexing all run for real.

Run with:
    pytest tests/test_build_sample_index.py -v
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.build_sample_index as build_mod  # noqa: E402


class _FakeEmbeddingModel:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        vectors = []
        for text in texts:
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            vectors.append(rng.rand(self.dim).astype(np.float32))
        return np.array(vectors)


def _make_synthetic_filtered_csv(path: Path, n_per_category=50):
    categories = ["Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]
    rows = []
    cid = 1
    for cat in categories:
        for i in range(n_per_category):
            rows.append({
                "Complaint ID": cid,
                "Date received": "2024-01-15",
                "product_category": cat,
                "Product": cat,
                "Sub-product": "",
                "Issue": "Billing dispute",
                "Sub-issue": "",
                "Company": "Test Bank",
                "State": "CA",
                "Consumer complaint narrative": f"raw narrative {cid} about {cat}",
                "cleaned_narrative": f"narrative {cid} {cat.lower().replace(' ', '_')} " * 15,
            })
            cid += 1
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


@pytest.fixture
def synthetic_csv(tmp_path):
    csv_path = tmp_path / "filtered_complaints.csv"
    _make_synthetic_filtered_csv(csv_path, n_per_category=50)
    return csv_path


def test_build_sample_index_end_to_end(monkeypatch, synthetic_csv, tmp_path):
    monkeypatch.setattr(build_mod, "load_embedding_model", lambda *a, **k: _FakeEmbeddingModel())

    output_dir = tmp_path / "vector_store_sample"
    manifest = build_mod.build_sample_index(
        input_path=str(synthetic_csv),
        output_dir=str(output_dir),
        collection_name="complaints_sample_test",
        sample_size=80,  # 40% of the 200-row synthetic set
        chunk_size=50,
        chunk_overlap=10,
        random_state=42,
    )

    assert manifest["sample_size_actual"] > 0
    assert manifest["num_chunks"] > 0
    assert (output_dir / "build_manifest.json").exists()


def test_build_sample_index_preserves_category_proportions(monkeypatch, synthetic_csv, tmp_path):
    monkeypatch.setattr(build_mod, "load_embedding_model", lambda *a, **k: _FakeEmbeddingModel())

    output_dir = tmp_path / "vector_store_sample"
    manifest = build_mod.build_sample_index(
        input_path=str(synthetic_csv),
        output_dir=str(output_dir),
        collection_name="complaints_sample_test",
        sample_size=80,
        random_state=42,
    )

    # synthetic data is perfectly balanced (50/category) -> sample should
    # stay roughly balanced too, not skew toward one category
    counts = list(manifest["category_counts"].values())
    assert max(counts) - min(counts) <= 5


def test_build_sample_index_manifest_is_valid_json(monkeypatch, synthetic_csv, tmp_path):
    monkeypatch.setattr(build_mod, "load_embedding_model", lambda *a, **k: _FakeEmbeddingModel())

    output_dir = tmp_path / "vector_store_sample"
    build_mod.build_sample_index(
        input_path=str(synthetic_csv),
        output_dir=str(output_dir),
        collection_name="complaints_sample_test",
        sample_size=80,
        random_state=42,
    )

    with open(output_dir / "build_manifest.json") as f:
        manifest = json.load(f)
    assert "sampled_complaint_ids" in manifest
    assert len(manifest["sampled_complaint_ids"]) == manifest["sample_size_actual"]


def test_build_sample_index_collection_is_queryable(monkeypatch, synthetic_csv, tmp_path):
    monkeypatch.setattr(build_mod, "load_embedding_model", lambda *a, **k: _FakeEmbeddingModel())

    output_dir = tmp_path / "vector_store_sample"
    build_mod.build_sample_index(
        input_path=str(synthetic_csv),
        output_dir=str(output_dir),
        collection_name="complaints_sample_test",
        sample_size=80,
        random_state=42,
    )

    import chromadb
    client = chromadb.PersistentClient(path=str(output_dir))
    collection = client.get_collection("complaints_sample_test")
    assert collection.count() > 0

    result = collection.query(
        query_embeddings=[_FakeEmbeddingModel().encode(["narrative about credit card"])[0].tolist()],
        n_results=3,
    )
    assert len(result["documents"][0]) == 3


def test_build_sample_index_reproducible_with_same_random_state(monkeypatch, synthetic_csv, tmp_path):
    monkeypatch.setattr(build_mod, "load_embedding_model", lambda *a, **k: _FakeEmbeddingModel())

    manifest_a = build_mod.build_sample_index(
        input_path=str(synthetic_csv),
        output_dir=str(tmp_path / "run_a"),
        collection_name="complaints_sample_test",
        sample_size=80,
        random_state=42,
    )
    manifest_b = build_mod.build_sample_index(
        input_path=str(synthetic_csv),
        output_dir=str(tmp_path / "run_b"),
        collection_name="complaints_sample_test",
        sample_size=80,
        random_state=42,
    )

    assert manifest_a["sampled_complaint_ids"] == manifest_b["sampled_complaint_ids"]