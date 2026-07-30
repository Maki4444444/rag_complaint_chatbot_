"""
Unit tests for the evaluation harness (src/evaluation.py).

These tests exercise the harness's own logic (question loading, judge
JSON parsing, aggregation, markdown rendering) with mocked LLM responses --
they never call the real HuggingFace API or a real vector store.

Run with:
    pytest tests/test_evaluation.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (  # noqa: E402
    load_questions,
    _extract_json,
    judge_answer,
    summarize,
    to_markdown_table,
    DEFAULT_QUESTIONS_PATH,
)


def test_load_questions_covers_all_four_categories():
    questions = load_questions(DEFAULT_QUESTIONS_PATH)
    categories = {q["category"] for q in questions}
    for required in ["Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]:
        assert required in categories


def test_load_questions_has_between_15_and_20_questions():
    questions = load_questions(DEFAULT_QUESTIONS_PATH)
    assert 15 <= len(questions) <= 20


def test_load_questions_each_has_required_fields():
    for q in load_questions(DEFAULT_QUESTIONS_PATH):
        assert q["id"] and q["category"] and q["question"]


def test_extract_json_parses_clean_json():
    raw = '{"retrieval_relevance": 4, "faithfulness": 5, "comment": "grounded"}'
    parsed = _extract_json(raw)
    assert parsed == {"retrieval_relevance": 4, "faithfulness": 5, "comment": "grounded"}


def test_extract_json_parses_json_wrapped_in_prose():
    raw = 'Sure, here is my score:\n{"retrieval_relevance": 3, "faithfulness": 2, "comment": "ok"}\nHope that helps!'
    parsed = _extract_json(raw)
    assert parsed["retrieval_relevance"] == 3


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


def test_judge_answer_returns_scores_on_success():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"retrieval_relevance": 5, "faithfulness": 4, "comment": "well grounded"}'
    )
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = mock_response

    result = judge_answer(mock_client, "q", "context", "answer")

    assert result["retrieval_relevance"] == 5
    assert result["faithfulness"] == 4
    assert result["judge_error"] is None


def test_judge_answer_handles_malformed_judge_response_gracefully():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "the answer looks good, I'd say a 4 and a 5"
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = mock_response

    result = judge_answer(mock_client, "q", "context", "answer")

    # Should not raise -- degrades to None scores with the error captured.
    assert result["retrieval_relevance"] is None
    assert result["faithfulness"] is None
    assert result["judge_error"] is not None


def test_judge_answer_handles_api_exception_gracefully():
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = TimeoutError("judge call timed out")

    result = judge_answer(mock_client, "q", "context", "answer")

    assert result["retrieval_relevance"] is None
    assert "timed out" in result["judge_error"]


def test_summarize_computes_expected_aggregates():
    results = {
        "num_questions": 3,
        "rows": [
            {"faithfulness": 5, "retrieval_relevance": 5},
            {"faithfulness": 4, "retrieval_relevance": 3},
            {"faithfulness": 2, "retrieval_relevance": 2},
        ],
    }
    summary = summarize(results)
    assert summary["n_scored"] == 3
    assert summary["avg_faithfulness"] == pytest.approx(11 / 3, rel=1e-3)
    # 2 of 3 have faithfulness >= 4
    assert summary["pct_faithful_ge_4"] == pytest.approx(2 / 3, rel=1e-3)
    assert summary["meets_faithfulness_target"] is False  # 0.667 < 0.85 target


def test_summarize_meets_target_when_faithfulness_is_high():
    results = {
        "num_questions": 4,
        "rows": [{"faithfulness": 5, "retrieval_relevance": 5} for _ in range(4)],
    }
    summary = summarize(results)
    assert summary["meets_faithfulness_target"] is True


def test_summarize_handles_all_judge_failures():
    results = {
        "num_questions": 2,
        "rows": [
            {"faithfulness": None, "retrieval_relevance": None},
            {"faithfulness": None, "retrieval_relevance": None},
        ],
    }
    summary = summarize(results)
    assert summary["n_scored"] == 0
    assert summary["meets_faithfulness_target"] is False


def test_to_markdown_table_renders_expected_columns():
    results = {
        "rows": [
            {
                "question": "Why are customers unhappy?",
                "answer": "Because of fees.",
                "retrieved_sources": [
                    {"product_category": "Credit Card", "complaint_id": "1", "similarity_score": 0.91}
                ],
                "retrieval_relevance": 5,
                "faithfulness": 4,
                "comment": "solid",
                "judge_error": None,
            }
        ]
    }
    table = to_markdown_table(results)
    assert "Question" in table and "Quality Score" in table
    assert "Why are customers unhappy?" in table
    assert "Credit Card #1" in table