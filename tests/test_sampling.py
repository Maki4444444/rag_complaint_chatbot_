"""
Unit tests for src/sampling.py.

Run with:
    pytest tests/test_sampling.py -v
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sampling import stratified_sample, summarize_sample  # noqa: E402


def _make_imbalanced_df(n_credit_card=500, n_personal_loan=100):
    rows = (
        [{"product_category": "Credit Card", "value": i} for i in range(n_credit_card)]
        + [{"product_category": "Personal Loan", "value": i} for i in range(n_personal_loan)]
    )
    return pd.DataFrame(rows)


def test_stratified_sample_returns_requested_size_approximately():
    df = _make_imbalanced_df(n_credit_card=500, n_personal_loan=100)
    sample = stratified_sample(df, n=120, random_state=42)
    # group-wise frac sampling rounds per-group, so allow small tolerance
    assert abs(len(sample) - 120) <= 5


def test_stratified_sample_preserves_category_proportions():
    df = _make_imbalanced_df(n_credit_card=500, n_personal_loan=100)  # 5:1 ratio
    sample = stratified_sample(df, n=120, random_state=42)

    counts = sample["product_category"].value_counts()
    ratio = counts["Credit Card"] / counts["Personal Loan"]
    # should stay close to the original 5:1 ratio, not become artificially balanced
    assert 3.5 <= ratio <= 6.5


def test_stratified_sample_includes_every_category():
    df = _make_imbalanced_df(n_credit_card=500, n_personal_loan=100)
    sample = stratified_sample(df, n=120, random_state=42)
    assert set(sample["product_category"].unique()) == {"Credit Card", "Personal Loan"}


def test_stratified_sample_is_reproducible_with_fixed_random_state():
    df = _make_imbalanced_df()
    sample_a = stratified_sample(df, n=120, random_state=42)
    sample_b = stratified_sample(df, n=120, random_state=42)
    assert sorted(sample_a["value"].tolist()) == sorted(sample_b["value"].tolist())


def test_stratified_sample_different_random_state_gives_different_rows():
    df = _make_imbalanced_df()
    sample_a = stratified_sample(df, n=120, random_state=42)
    sample_b = stratified_sample(df, n=120, random_state=99)
    assert sorted(sample_a["value"].tolist()) != sorted(sample_b["value"].tolist())


def test_stratified_sample_n_greater_than_df_returns_full_df():
    df = _make_imbalanced_df(n_credit_card=10, n_personal_loan=5)
    sample = stratified_sample(df, n=1000, random_state=42)
    assert len(sample) == len(df)


def test_summarize_sample_counts_and_percentages_sum_correctly():
    df = _make_imbalanced_df(n_credit_card=80, n_personal_loan=20)
    summary = summarize_sample(df)

    assert summary.loc["Credit Card", "count"] == 80
    assert summary.loc["Personal Loan", "count"] == 20
    assert abs(summary["pct"].sum() - 100.0) < 0.1


def test_summarize_sample_sorted_descending_by_count():
    df = _make_imbalanced_df(n_credit_card=80, n_personal_loan=20)
    summary = summarize_sample(df)
    assert summary.index.tolist()[0] == "Credit Card"