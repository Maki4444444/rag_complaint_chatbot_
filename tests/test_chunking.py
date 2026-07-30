"""
Unit tests for src/chunking.py.

--------------------------------------------------------------------------
IMPROVEMENT OVER v1: no tests existed for chunking.py at all in v1 (only
preprocessing.py had a test file). This closes that gap.
--------------------------------------------------------------------------

Run with:
    pytest tests/test_chunking.py -v
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import get_text_splitter, chunk_dataframe  # noqa: E402


def test_get_text_splitter_default_params():
    splitter = get_text_splitter()
    assert splitter._chunk_size == 500
    assert splitter._chunk_overlap == 50


def test_get_text_splitter_custom_params():
    splitter = get_text_splitter(chunk_size=100, chunk_overlap=10)
    assert splitter._chunk_size == 100
    assert splitter._chunk_overlap == 10


def test_chunk_dataframe_splits_long_text():
    df = pd.DataFrame({
        "Complaint ID": [1],
        "product_category": ["Credit Card"],
        "cleaned_narrative": ["word " * 200],  # long enough to force multiple chunks
    })
    splitter = get_text_splitter(chunk_size=100, chunk_overlap=10)
    records = chunk_dataframe(df, text_column="cleaned_narrative", splitter=splitter)

    assert len(records) > 1
    assert all(r["complaint_id"] == 1 for r in records)
    assert all(r["product_category"] == "Credit Card" for r in records)


def test_chunk_dataframe_short_text_produces_one_chunk():
    df = pd.DataFrame({
        "Complaint ID": [1],
        "product_category": ["Credit Card"],
        "cleaned_narrative": ["short complaint text"],
    })
    splitter = get_text_splitter(chunk_size=500, chunk_overlap=50)
    records = chunk_dataframe(df, text_column="cleaned_narrative", splitter=splitter)

    assert len(records) == 1
    assert records[0]["chunk_index"] == 0
    assert records[0]["total_chunks"] == 1


def test_chunk_dataframe_sets_correct_chunk_index_and_total():
    df = pd.DataFrame({
        "Complaint ID": [1],
        "product_category": ["Credit Card"],
        "cleaned_narrative": ["word " * 200],
    })
    splitter = get_text_splitter(chunk_size=100, chunk_overlap=10)
    records = chunk_dataframe(df, text_column="cleaned_narrative", splitter=splitter)

    total = records[0]["total_chunks"]
    assert all(r["total_chunks"] == total for r in records)
    assert [r["chunk_index"] for r in records] == list(range(total))


def test_chunk_dataframe_skips_empty_narratives():
    df = pd.DataFrame({
        "Complaint ID": [1, 2],
        "product_category": ["Credit Card", "Personal Loan"],
        "cleaned_narrative": ["", "a valid complaint narrative here"],
    })
    splitter = get_text_splitter()
    records = chunk_dataframe(df, text_column="cleaned_narrative", splitter=splitter)

    assert all(r["complaint_id"] != 1 for r in records)
    assert any(r["complaint_id"] == 2 for r in records)


def test_chunk_dataframe_skips_non_string_narratives():
    df = pd.DataFrame({
        "Complaint ID": [1, 2],
        "product_category": ["Credit Card", "Personal Loan"],
        "cleaned_narrative": [None, "a valid complaint narrative here"],
    })
    splitter = get_text_splitter()
    records = chunk_dataframe(df, text_column="cleaned_narrative", splitter=splitter)

    assert all(r["complaint_id"] != 1 for r in records)


def test_chunk_dataframe_multiple_complaints_traceable_by_id():
    df = pd.DataFrame({
        "Complaint ID": [10, 20],
        "product_category": ["Credit Card", "Money Transfer"],
        "cleaned_narrative": ["first complaint text here", "second complaint text here"],
    })
    splitter = get_text_splitter()
    records = chunk_dataframe(df, text_column="cleaned_narrative", splitter=splitter)

    ids_seen = {r["complaint_id"] for r in records}
    assert ids_seen == {10, 20}


def test_chunk_dataframe_returns_empty_list_for_empty_dataframe():
    df = pd.DataFrame({"Complaint ID": [], "product_category": [], "cleaned_narrative": []})
    splitter = get_text_splitter()
    records = chunk_dataframe(df, text_column="cleaned_narrative", splitter=splitter)
    assert records == []