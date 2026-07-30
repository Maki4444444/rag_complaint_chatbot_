"""
Unit tests for the runtime-resilience pass (src/retriever.py, src/generator.py).

Covers the three failure modes called out in the plan:
    1. LLM API timeout/error          -> src/generator.py
    2. Empty retrieval results        -> src/generator.py (short-circuit)
    3. Malformed/empty user input     -> src/retriever.py (validate_question)

Run with:
    pytest tests/test_error_handling.py -v
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever import (  # noqa: E402
    validate_question,
    build_retriever,
    RetrievalError,
    VALID_CATEGORIES,
)
from src.generator import (  # noqa: E402
    generate_answer,
    is_transient_error,
    NO_CONTEXT_MESSAGE,
    LLM_FAILURE_MESSAGE,
    MAX_RETRIES,
)


# ── 1. Malformed / empty input (src/retriever.py) ───────────────────────────

def test_validate_question_rejects_empty_string():
    with pytest.raises(ValueError):
        validate_question("")


def test_validate_question_rejects_whitespace_only():
    with pytest.raises(ValueError):
        validate_question("   \n\t  ")


def test_validate_question_rejects_non_string():
    with pytest.raises(ValueError):
        validate_question(None)
    with pytest.raises(ValueError):
        validate_question(12345)


def test_validate_question_rejects_overlong_input():
    with pytest.raises(ValueError):
        validate_question("a" * 5000)


def test_validate_question_strips_and_accepts_valid_input():
    assert validate_question("  Why are customers unhappy?  ") == "Why are customers unhappy?"


def test_retrieve_rejects_invalid_category():
    mock_collection = MagicMock()
    mock_model = MagicMock()
    retrieve = build_retriever(mock_collection, model=mock_model)

    with pytest.raises(ValueError):
        retrieve("a valid question", product_category="Not A Real Category")

    # Should fail validation before ever touching the model or the store.
    mock_model.encode.assert_not_called()
    mock_collection.query.assert_not_called()


def test_retrieve_rejects_empty_question_before_embedding():
    mock_collection = MagicMock()
    mock_model = MagicMock()
    retrieve = build_retriever(mock_collection, model=mock_model)

    with pytest.raises(ValueError):
        retrieve("   ")

    mock_model.encode.assert_not_called()


# ── 2. Vector store / retrieval failure (src/retriever.py) ──────────────────

def test_retrieve_wraps_store_failure_in_retrieval_error():
    mock_collection = MagicMock()
    mock_collection.query.side_effect = RuntimeError("connection reset")
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    retrieve = build_retriever(mock_collection, model=mock_model)

    with pytest.raises(RetrievalError):
        retrieve("Why are customers unhappy with credit cards?")


def test_retrieve_returns_empty_list_on_legitimate_zero_results():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    retrieve = build_retriever(mock_collection, model=mock_model)
    result = retrieve("a question about something not in the dataset")

    assert result == []


# ── 3. Empty retrieval results (src/generator.py short-circuit) ─────────────

def test_generate_answer_short_circuits_on_empty_chunks():
    mock_client = MagicMock()

    result = generate_answer(mock_client, "some question", retrieved_chunks=[])

    assert result["answer"] == NO_CONTEXT_MESSAGE
    assert result["sources"] == []
    assert result["error"] is False
    # The LLM must never be called when there's no context to ground it.
    mock_client.chat_completion.assert_not_called()


# ── 4. LLM API failure / timeout (src/generator.py retry + fallback) ────────

SAMPLE_CHUNKS = [
    {
        "chunk_text": "My credit card was charged twice for the same purchase.",
        "complaint_id": "12345",
        "product_category": "Credit Card",
        "chunk_index": 0,
        "total_chunks": 1,
        "similarity_score": 0.87,
    }
]


def test_is_transient_error_classifies_timeout_as_retryable():
    assert is_transient_error(TimeoutError("timed out")) is True


def test_is_transient_error_classifies_connection_error_as_retryable():
    assert is_transient_error(ConnectionError("connection refused")) is True


def test_is_transient_error_classifies_type_error_as_non_retryable():
    # A programming error should never trigger a retry loop.
    assert is_transient_error(TypeError("bad argument")) is False


def test_generate_answer_returns_fallback_after_exhausting_retries(monkeypatch):
    # Avoid real sleeping during the retry backoff in tests.
    monkeypatch.setattr(time, "sleep", lambda _: None)

    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = TimeoutError("simulated API timeout")

    result = generate_answer(mock_client, "Why are customers unhappy?", SAMPLE_CHUNKS)

    assert result["error"] is True
    assert result["answer"] == LLM_FAILURE_MESSAGE
    # Retried MAX_RETRIES times total, not once and not indefinitely.
    assert mock_client.chat_completion.call_count == MAX_RETRIES


def test_generate_answer_succeeds_after_transient_failure_then_recovery(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Customers report duplicate charges."

    mock_client = MagicMock()
    # First call times out, second call succeeds.
    mock_client.chat_completion.side_effect = [
        TimeoutError("simulated transient failure"),
        mock_response,
    ]

    result = generate_answer(mock_client, "Why are customers unhappy?", SAMPLE_CHUNKS)

    assert result["error"] is False
    assert result["answer"] == "Customers report duplicate charges."
    assert mock_client.chat_completion.call_count == 2


def test_generate_answer_does_not_retry_non_transient_error(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)

    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = TypeError("programming error, not transient")

    result = generate_answer(mock_client, "Why are customers unhappy?", SAMPLE_CHUNKS)

    assert result["error"] is True
    # Fails fast on the first attempt -- no point retrying a bug.
    assert mock_client.chat_completion.call_count == 1


# ── 5. Rich metadata fields (full store) vs. sparse (sample store) ──────────
# The full store (vector_store_full/, built by build_full_index.py) carries
# company/issue/sub_issue/date_received; the Task 2 sample store
# (vector_store_sample/, built by build_sample_index.py) doesn't. retrieve()
# must work against either without raising.

def test_retrieve_includes_rich_metadata_fields_when_present():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["some chunk text"]],
        "metadatas": [[{
            "complaint_id": "123",
            "product_category": "Credit Card",
            "chunk_index": 0,
            "total_chunks": 1,
            "company": "Test Bank",
            "issue": "Billing dispute",
            "sub_issue": "Other problem",
            "date_received": "2025-06-13",
        }]],
        "distances": [[0.2]],
    }
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    retrieve = build_retriever(mock_collection, model=mock_model)
    result = retrieve("why are customers unhappy")

    assert result[0]["company"] == "Test Bank"
    assert result[0]["issue"] == "Billing dispute"
    assert result[0]["sub_issue"] == "Other problem"
    assert result[0]["date_received"] == "2025-06-13"


def test_retrieve_defaults_rich_metadata_fields_when_absent():
    # Simulates querying the Task 2 sample store, whose metadata schema
    # doesn't include company/issue/sub_issue/date_received at all.
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["some chunk text"]],
        "metadatas": [[{
            "complaint_id": "123",
            "product_category": "Credit Card",
            "chunk_index": 0,
            "total_chunks": 1,
        }]],
        "distances": [[0.2]],
    }
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    retrieve = build_retriever(mock_collection, model=mock_model)
    result = retrieve("why are customers unhappy")

    assert result[0]["company"] == ""
    assert result[0]["issue"] == ""
    assert result[0]["sub_issue"] == ""
    assert result[0]["date_received"] == ""
    # core fields still present and correct
    assert result[0]["complaint_id"] == "123"