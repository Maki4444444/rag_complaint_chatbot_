"""
Unit tests for the preprocessing module (src/preprocessing.py).

Run with:
    pytest tests/test_preprocessing.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (  # noqa: E402
    clean_text,
    clean_text_noise,
    normalize_text,
    strip_boilerplate_openers,
    build_reverse_product_map,
    get_target_raw_products,
    PRODUCT_MAP,
)


# --- carried over from v1 --------------------------------------------------

def test_clean_text_noise_lowercases():
    assert "hello world" in clean_text_noise("HELLO World")


def test_clean_text_noise_removes_urls():
    result = clean_text_noise("Visit our site: http://example.com now")
    assert "http" not in result
    assert "example.com" not in result


def test_clean_text_noise_removes_phone_numbers():
    result = clean_text_noise("Call me at 555-1234 please")
    assert "555" not in result


def test_clean_text_noise_removes_html_tags():
    result = clean_text_noise("I hate the <br>high fees")
    assert "<br>" not in result
    assert "high fees" in result


def test_clean_text_noise_removes_redaction_placeholders():
    result = clean_text_noise("My account XXXX was charged on XX/XX/2024")
    assert "xxxx" not in result


def test_clean_text_noise_removes_special_characters():
    result = clean_text_noise("Fee of $50.00!! Why??")
    assert "$" not in result
    assert "!" not in result


def test_clean_text_noise_handles_empty_string():
    assert clean_text_noise("") == ""


def test_clean_text_noise_handles_none_gracefully():
    assert clean_text_noise(None) == ""


def test_clean_text_noise_handles_nan_like_float():
    assert clean_text_noise(float("nan")) == ""


def test_normalize_text_removes_stopwords():
    result = normalize_text("this is a complaint about the credit card")
    assert "is" not in result.split()
    assert "the" not in result.split()


def test_normalize_text_lemmatizes_verbs():
    result = normalize_text("the app keeps crashing")
    assert "crash" in result


def test_normalize_text_handles_empty_string():
    assert normalize_text("") == ""


def test_build_reverse_product_map_covers_all_categories():
    reverse_map = build_reverse_product_map()
    assert set(reverse_map.values()) == set(PRODUCT_MAP.keys())


def test_get_target_raw_products_matches_product_map():
    raw_products = get_target_raw_products()
    expected_count = sum(len(v) for v in PRODUCT_MAP.values())
    assert len(raw_products) == expected_count
    assert "Credit card" in raw_products


# --- new: boilerplate-opener stripping (closes v1 gap) ---------------------

def test_strip_boilerplate_i_am_writing_to_file():
    result = strip_boilerplate_openers(
        "i am writing to file a complaint about my credit card. it was charged twice."
    )
    assert "writing to file a complaint" not in result
    assert "charged twice" in result


def test_strip_boilerplate_im_contraction():
    result = strip_boilerplate_openers(
        "i'm writing to file a complaint regarding fraud. my account was hacked."
    )
    assert "writing to file a complaint" not in result
    assert "account was hacked" in result


def test_strip_boilerplate_submit_variant():
    result = strip_boilerplate_openers(
        "i am writing to submit a complaint about overdraft fees. this is unfair."
    )
    assert "writing to submit a complaint" not in result
    assert "unfair" in result


def test_strip_boilerplate_to_whom_it_may_concern():
    result = strip_boilerplate_openers(
        "to whom it may concern, my loan was denied without explanation."
    )
    assert "to whom it may concern" not in result
    assert "loan was denied" in result


def test_strip_boilerplate_writing_to_express():
    result = strip_boilerplate_openers(
        "i am writing to express my frustration with your bank. i was overcharged."
    )
    assert "writing to express" not in result
    assert "overcharged" in result


def test_strip_boilerplate_would_like_to_file():
    result = strip_boilerplate_openers(
        "i would like to file a complaint about identity theft on my account."
    )
    assert "would like to file a complaint" not in result
    assert "identity theft" in result


def test_strip_boilerplate_does_not_remove_midsentence_mentions():
    # "complaint" appearing later in the narrative, not as an opener,
    # should be left alone -- this guards against an overly broad filter.
    result = strip_boilerplate_openers(
        "the bank told me to file a complaint but never followed up."
    )
    assert "file a complaint" in result


def test_strip_boilerplate_leaves_non_boilerplate_text_untouched():
    text = "my credit card was declined at checkout for no reason."
    assert strip_boilerplate_openers(text) == text


def test_strip_boilerplate_handles_empty_string():
    assert strip_boilerplate_openers("") == ""


def test_strip_boilerplate_handles_stacked_openers():
    # salutation immediately followed by a second formulaic opener
    result = strip_boilerplate_openers(
        "to whom it may concern, i am writing to file a complaint about fraud on my account."
    )
    assert "to whom it may concern" not in result
    assert "writing to file a complaint" not in result
    assert "fraud on my account" in result


def test_clean_text_noise_integrates_boilerplate_stripping():
    result = clean_text_noise(
        "I am writing to file a complaint! My account XXXX was charged twice. Call 555-1234."
    )
    assert "writing to file a complaint" not in result
    assert "charged twice" in result
    assert "xxxx" not in result
    assert "555" not in result


def test_clean_text_full_pipeline_strips_boilerplate_and_normalizes():
    text = "I am writing to file a complaint! My account XXXX was crashing. Call 555-1234."
    result = clean_text(text)
    assert "writing" not in result.split() or "file" not in result.split()
    assert "xxxx" not in result
    assert "555" not in result
    assert "crash" in result  # lemmatized form of "crashing"