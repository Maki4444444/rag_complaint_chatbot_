"""
Integration test for src/build_full_index.py -- builds a synthetic
parquet file matching the EXACT schema confirmed against the real
complaint_embeddings.parquet (via src/explore_parquet.py output: id,
document, embedding<list<double,384>>, metadata<struct>), then runs the
real ingestion pipeline against it. No mocking of chromadb or pyarrow --
both are fast and local enough to test for real.

Run with:
    pytest tests/test_build_full_index.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.build_full_index as build_mod  # noqa: E402


def _make_synthetic_parquet(path: Path, n_rows: int = 500, n_row_groups: int = 2):
    """
    Builds a parquet file with the same Arrow schema as the real
    complaint_embeddings.parquet: top-level id/document/embedding/metadata
    columns, metadata as a struct (not flattened columns), embedding as a
    384-dim double list -- matching src/explore_parquet.py's confirmed
    output against the real file.
    """
    categories = ["Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]
    products = {
        "Credit Card": "Credit card",
        "Personal Loan": "Payday loan, title loan, or personal loan",
        "Savings Account": "Checking or savings account",
        "Money Transfer": "Money transfers",
    }
    rng = np.random.RandomState(42)

    ids, documents, embeddings, metadatas = [], [], [], []
    for i in range(n_rows):
        complaint_id = str(10_000_000 + i // 2)  # some complaints get 2 chunks
        chunk_index = i % 2
        total_chunks = 2 if i % 4 < 2 else 1
        category = categories[i % len(categories)]

        ids.append(f"{complaint_id}_{chunk_index}")
        documents.append(f"synthetic complaint narrative chunk {i} about {category.lower()}")
        embeddings.append(rng.rand(384).astype(np.float64).tolist())
        metadatas.append({
            "chunk_index": chunk_index,
            "company": "Test Bank",
            "complaint_id": complaint_id,
            "date_received": "2025-06-13",
            "issue": "Billing dispute",
            "product": products[category],
            "product_category": category,
            "state": "TX",
            "sub_issue": "Other problem",
            "total_chunks": total_chunks,
        })

    table = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "document": pa.array(documents, type=pa.string()),
        "embedding": pa.array(embeddings, type=pa.list_(pa.float64())),
        "metadata": pa.array(metadatas, type=pa.struct([
            ("chunk_index", pa.int64()),
            ("company", pa.string()),
            ("complaint_id", pa.string()),
            ("date_received", pa.string()),
            ("issue", pa.string()),
            ("product", pa.string()),
            ("product_category", pa.string()),
            ("state", pa.string()),
            ("sub_issue", pa.string()),
            ("total_chunks", pa.int64()),
        ])),
    })

    # Write with multiple row groups, matching the real file's structure,
    # to exercise iter_batches() spanning row-group boundaries.
    rows_per_group = max(1, n_rows // n_row_groups)
    pq.write_table(table, path, row_group_size=rows_per_group)
    return table


@pytest.fixture
def synthetic_parquet(tmp_path):
    path = tmp_path / "complaint_embeddings.parquet"
    _make_synthetic_parquet(path, n_rows=500, n_row_groups=3)
    return path


def test_build_full_index_ingests_all_rows(synthetic_parquet, tmp_path):
    output_dir = tmp_path / "vector_store_full"
    manifest = build_mod.build_full_index(
        input_path=str(synthetic_parquet),
        output_dir=str(output_dir),
        collection_name="complaints_full_test",
        read_batch_size=100,  # smaller than the file, forces multiple batches
    )

    assert manifest["rows_in_source_file"] == 500
    assert manifest["rows_ingested"] == 500
    assert manifest["final_collection_count"] == 500


def test_build_full_index_manifest_written(synthetic_parquet, tmp_path):
    output_dir = tmp_path / "vector_store_full"
    build_mod.build_full_index(
        input_path=str(synthetic_parquet),
        output_dir=str(output_dir),
        collection_name="complaints_full_test",
        read_batch_size=100,
    )

    manifest_path = output_dir / "build_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["collection_name"] == "complaints_full_test"


def test_build_full_index_category_counts_sum_to_total(synthetic_parquet, tmp_path):
    output_dir = tmp_path / "vector_store_full"
    manifest = build_mod.build_full_index(
        input_path=str(synthetic_parquet),
        output_dir=str(output_dir),
        collection_name="complaints_full_test",
        read_batch_size=100,
    )

    assert sum(manifest["category_counts"].values()) == manifest["rows_ingested"]
    assert set(manifest["category_counts"].keys()) == {
        "Credit Card", "Personal Loan", "Savings Account", "Money Transfer",
    }


def test_build_full_index_preserves_rich_metadata_fields(synthetic_parquet, tmp_path):
    # This is the field set the real parquet has that the Task 2 sample
    # store does NOT (company, date_received, issue, sub_issue) --
    # confirming these survive ingestion, not just the minimal fields.
    output_dir = tmp_path / "vector_store_full"
    build_mod.build_full_index(
        input_path=str(synthetic_parquet),
        output_dir=str(output_dir),
        collection_name="complaints_full_test",
        read_batch_size=100,
    )

    import chromadb
    client = chromadb.PersistentClient(path=str(output_dir))
    collection = client.get_collection("complaints_full_test")
    result = collection.get(ids=["10000000_0"], include=["metadatas", "documents"])

    meta = result["metadatas"][0]
    assert meta["company"] == "Test Bank"
    assert meta["date_received"] == "2025-06-13"
    assert meta["issue"] == "Billing dispute"
    assert meta["sub_issue"] == "Other problem"
    assert meta["state"] == "TX"
    assert meta["chunk_index"] == 0
    assert isinstance(meta["chunk_index"], int)


def test_build_full_index_is_idempotent_on_rerun(synthetic_parquet, tmp_path):
    output_dir = tmp_path / "vector_store_full"
    build_mod.build_full_index(
        input_path=str(synthetic_parquet), output_dir=str(output_dir),
        collection_name="complaints_full_test", read_batch_size=100,
    )
    manifest_2 = build_mod.build_full_index(
        input_path=str(synthetic_parquet), output_dir=str(output_dir),
        collection_name="complaints_full_test", read_batch_size=100,
    )
    assert manifest_2["final_collection_count"] == 500  # not doubled


def test_build_full_index_respects_limit_rows(synthetic_parquet, tmp_path):
    output_dir = tmp_path / "vector_store_full"
    manifest = build_mod.build_full_index(
        input_path=str(synthetic_parquet),
        output_dir=str(output_dir),
        collection_name="complaints_full_test",
        read_batch_size=100,
        limit_rows=150,
    )
    assert manifest["rows_ingested"] == 150
    assert manifest["final_collection_count"] == 150


def test_build_full_index_collection_is_queryable(synthetic_parquet, tmp_path):
    output_dir = tmp_path / "vector_store_full"
    build_mod.build_full_index(
        input_path=str(synthetic_parquet), output_dir=str(output_dir),
        collection_name="complaints_full_test", read_batch_size=100,
    )

    import chromadb
    client = chromadb.PersistentClient(path=str(output_dir))
    collection = client.get_collection("complaints_full_test")

    query_embedding = np.random.RandomState(0).rand(384).tolist()
    result = collection.query(query_embeddings=[query_embedding], n_results=5)
    assert len(result["documents"][0]) == 5


def test_build_full_index_spans_row_group_boundaries(tmp_path):
    # Read batch size smaller than a single row group AND smaller than
    # total rows, to force iter_batches() across multiple row groups and
    # multiple read batches within a group -- the real file has 2 row
    # groups covering 1.37M rows, so this must not silently drop or
    # duplicate rows at the boundary.
    path = tmp_path / "complaint_embeddings.parquet"
    _make_synthetic_parquet(path, n_rows=1000, n_row_groups=4)

    output_dir = tmp_path / "vector_store_full"
    manifest = build_mod.build_full_index(
        input_path=str(path), output_dir=str(output_dir),
        collection_name="complaints_full_test", read_batch_size=37,  # deliberately awkward
    )
    assert manifest["rows_ingested"] == 1000
    assert manifest["final_collection_count"] == 1000


def test_flatten_metadata_coerces_int_fields():
    flat = build_mod._flatten_metadata({
        "chunk_index": 3, "total_chunks": 5, "company": "Test Bank", "state": None,
    })
    assert flat["chunk_index"] == 3
    assert isinstance(flat["chunk_index"], int)
    assert flat["state"] == ""  # None coerced to empty string, not "None"


def test_flatten_metadata_handles_missing_optional_field():
    # A field being absent from one row's dict (not just None) shouldn't crash.
    flat = build_mod._flatten_metadata({"complaint_id": "123", "chunk_index": 0})
    assert flat["complaint_id"] == "123"
    assert flat["chunk_index"] == 0