"""
src/sampling.py

Stratified sampling for Task 2: draws a 10K-15K complaint sample from the
cleaned/filtered dataset, proportionally representative of each product
category.

--------------------------------------------------------------------------
IMPROVEMENT OVER v1 (Maki4444444/rag-complaint-chatbot):
v1's README documents a 12,500-complaint sample and reports the resulting
per-category counts, but no sampling code was ever committed to the
repo -- notebooks/README.md is empty and there's no sampling script in
src/. That sample store still exists as a runtime artifact
(vector_store/), but nothing in the repository can reproduce it. This
module exists so the sampling step has a name, a fixed random_state, and
a test, the same way strip_boilerplate_openers() closed the analogous
gap in Task 1. See docs/WHY_A_CLEAN_REBUILD.md.
--------------------------------------------------------------------------
"""
import pandas as pd


def stratified_sample(
    df: pd.DataFrame,
    n: int = 12_500,
    category_col: str = "product_category",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Draws a stratified sample of size `n` from `df`, proportional to each
    category's share of the full (filtered) dataset.

    Why proportional rather than equal-per-category sampling: the Task 1
    EDA found the four categories are naturally imbalanced (~5:1 between
    Credit Card and Personal Loan). An equal-per-category sample would
    force Personal Loan up to the same count as Credit Card, i.e.
    synthetically inflate a category that's genuinely smaller in the
    real complaint volume CrediTrust receives -- misleading for a system
    meant to reflect actual complaint patterns. Proportional sampling
    keeps that real-world imbalance intact at 1/Nth scale instead of
    manufacturing balance that doesn't exist in production.

    Why a fixed random_state: this sample is meant to be reproducible --
    re-running this function should draw the same rows so that Task 2
    isn't silently non-deterministic between runs.

    Uses pandas' group-wise .sample(frac=...) rather than a single
    global .sample(n=...), specifically so every category is
    represented (a single global sample could, by chance, under- or
    zero-represent the smallest category).
    """
    if n >= len(df):
        return df.copy()

    frac = n / len(df)

    sampled = (
        df.groupby(category_col, observed=True)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )

    return sampled


def summarize_sample(df: pd.DataFrame, category_col: str = "product_category") -> pd.DataFrame:
    """
    Returns a per-category count + percentage table, for logging/
    reporting the sample's composition (used by build_sample_index.py's
    printed summary and can be pasted straight into the report's
    "sampling strategy" section).
    """
    counts = df[category_col].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"count": counts, "pct": pct}).sort_values("count", ascending=False)