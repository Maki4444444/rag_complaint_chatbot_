"""
src/preprocessing.py

Text cleaning and product-category mapping utilities for the CFPB complaint
dataset (Task 1: EDA & Preprocessing), reused by Task 2 (chunking).

Cleaning pipeline, in order:
  1. strip_boilerplate_openers() -> remove stock complaint-letter openers
                                     ("I am writing to file a complaint...")
  2. clean_text_noise()          -> lowercase, strip URLs/phone numbers/
                                     IDs/HTML/punctuation
  3. normalize_text()            -> tokenize, remove stopwords, lemmatize

--------------------------------------------------------------------------
IMPROVEMENT OVER v1 (Maki4444444/rag-complaint-chatbot):
The Task 1 instructions name boilerplate-opener stripping as an explicit
example cleaning step ("e.g. 'I am writing to file a complaint...'"). The
v1 pipeline never implemented it — clean_text_noise() only ever stripped
URLs/phone numbers/HTML/punctuation, not stock opening sentences. Left in
the narrative, these openers are near-identical across thousands of
complaints and add no retrieval signal, so they dilute chunk embeddings
with boilerplate rather than complaint-specific content. strip_boilerplate_
openers() below is written as its own testable function (not folded
silently into clean_text_noise) specifically so this requirement has a
name, a docstring, and a dedicated test file, rather than being an
implicit side effect someone has to infer from reading regexes.
See docs/WHY_A_CLEAN_REBUILD.md for the full list of what this rebuild
changes vs. v1.
--------------------------------------------------------------------------
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK corpora (no-ops if already present).
for _resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        pass

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Maps our 4 target product categories to the raw CFPB `Product` column
# values that should be considered part of that category. CFPB has renamed
# these labels over the years, so multiple raw values can map to one category.
PRODUCT_MAP = {
    "Credit Card": [
        "Credit card",
        "Credit card or prepaid card",
    ],
    "Personal Loan": [
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan",
        "Consumer Loan",
        "Payday loan",
    ],
    "Savings Account": [
        "Bank account or service",
        "Checking or savings account",
    ],
    "Money Transfer": [
        "Money transfers",
        "Money transfer, virtual currency, or money service",
    ],
}

# ---------------------------------------------------------------------------
# Boilerplate-opener patterns.
#
# Why regex-at-start rather than a keyword blocklist: these openers are
# formulaic (consumer complaint forms and template letters produce very
# similar first sentences), but wording varies enough ("file"/"submit"/
# "lodge a complaint", "I am"/"I'm") that a single fixed phrase would miss
# most instances. Anchoring each pattern to the *start* of the narrative
# (^) is deliberate: these phrases are complaint-letter salutations, not
# something we want to strip if they happen to appear mid-narrative (e.g.
# a consumer quoting a company's own boilerplate back at them is still
# meaningful content).
#
# Each pattern matches only the fixed opener phrase itself, plus at most
# one immediate connector word ("about"/"regarding"/"concerning") if it
# directly follows -- NOT an open-ended run up to the next punctuation.
# An earlier version tried to consume "everything up to the next sentence
# delimiter", which silently swallowed the entire narrative whenever the
# opener ran straight into the complaint with no punctuation in between
# (e.g. "i would like to file a complaint about identity theft..." has no
# period until the very end of the sentence). Matching only the fixed
# phrase avoids that failure mode entirely.
#
# Applied case-insensitively, before punctuation stripping, so patterns
# ending in "[,:]?" can still match a trailing comma/colon and get removed
# cleanly rather than leaving a dangling ", my account was charged...".
# ---------------------------------------------------------------------------
_BOILERPLATE_OPENER_PATTERNS = [
    r"^\s*i\s*(?:'m|am)\s+writing\s+to\s+(?:file|submit|lodge|report)\s+a\s+complaint\b"
    r"(?:\s+(?:about|regarding|concerning))?\s*",
    r"^\s*i\s+am\s+writing\s+(?:this\s+letter\s+)?to\s+(?:express|inform|report|notify)\b"
    r"(?:\s+(?:you|my|about|regarding|concerning))?\s*",
    r"^\s*i\s+would\s+like\s+to\s+file\s+a\s+complaint\b"
    r"(?:\s+(?:about|regarding|concerning))?\s*",
    r"^\s*this\s+letter\s+is\s+to\s+(?:inform|notify|report)\b"
    r"(?:\s+(?:you|about|regarding|concerning))?\s*",
    r"^\s*to\s+whom\s+it\s+may\s+concern[,:]?\s*",
    r"^\s*i\s+am\s+filing\s+(?:this\s+)?complaint\s+against\b\s*",
]


def strip_boilerplate_openers(text: str) -> str:
    """
    Removes stock complaint-letter opening phrases from the start of a
    narrative (e.g. "I am writing to file a complaint about...",
    "To whom it may concern,").

    Only strips at the start of the string, and only known formulaic
    openers — this is intentionally conservative rather than a broad
    "remove any sentence containing 'complaint'" filter, since the word
    "complaint" itself is often meaningful complaint-specific content.

    Runs in a loop (bounded) because some narratives stack more than one
    opener back to back (e.g. a salutation followed by a "writing to
    file..." sentence).

    Expects lowercased input (called after the lowercase step in
    clean_text_noise). Returns an empty string for empty input.
    """
    if not text:
        return ""

    cleaned = text
    for _ in range(3):  # bounded loop guards against pathological input
        before = cleaned
        for pattern in _BOILERPLATE_OPENER_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
        if cleaned == before:
            break

    return cleaned.strip()


def clean_text_noise(text) -> str:
    """
    Step 1 cleaning: lowercase, strip boilerplate openers, then strip
    noise (URLs, phone numbers, simple IDs/IBANs, HTML tags, punctuation/
    special characters).

    Returns an empty string for None / NaN-like / empty input.
    """
    if text is None:
        return ""
    text = str(text)
    if text.strip() == "" or text.lower() == "nan":
        return ""

    text = text.lower()

    # Strip stock complaint-letter openers before anything else touches
    # punctuation, since the patterns rely on trailing commas/periods to
    # anchor cleanly.
    text = strip_boilerplate_openers(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove phone numbers (7-digit and 10-digit formats)
    text = re.sub(r"\b\d{3}[-.\s]?\d{4}\b", "", text)
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "", text)

    # Remove simple IDs/IBANs (2 letters followed by 10+ digits)
    text = re.sub(r"\b[a-z]{2}\d{10,}\b", "", text, flags=re.I)

    # Remove CFPB redaction placeholders, e.g. "xxxx" / "xx/xx/xxxx"
    text = re.sub(r"x{2,}", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and special characters (keep word chars + spaces)
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_text(text: str) -> str:
    """
    Step 2 cleaning: tokenize, remove English stopwords, and lemmatize
    (both as verbs and nouns) so that inflected forms collapse to a
    common root (e.g. "crashing"/"crashed" -> "crash"). This goes beyond
    what Task 1 strictly requires ("apply additional normalization as
    appropriate") but reduces vocabulary sparsity ahead of Task 2's
    embedding step, where "crash"/"crashing"/"crashed" should ideally
    embed close together rather than as unrelated tokens.

    Expects already-noise-cleaned text (see clean_text_noise). Returns
    an empty string for empty input.
    """
    if not text:
        return ""

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOP_WORDS]

    lemmas = [_LEMMATIZER.lemmatize(t, pos="v") for t in tokens]
    lemmas = [_LEMMATIZER.lemmatize(t, pos="n") for t in lemmas]

    return " ".join(lemmas)


def clean_text(text) -> str:
    """
    Full cleaning pipeline: clean_text_noise() (includes boilerplate-
    opener stripping) followed by normalize_text(). This is the single
    entry point used by the EDA notebook and Task 2.
    """
    noise_cleaned = clean_text_noise(text)
    return normalize_text(noise_cleaned)


def build_reverse_product_map(product_map: dict = PRODUCT_MAP) -> dict:
    """
    Flatten PRODUCT_MAP into {raw_product_label: clean_category_name}
    for use with df["Product"].map(...).
    """
    return {raw: clean for clean, raws in product_map.items() for raw in raws}


def get_target_raw_products(product_map: dict = PRODUCT_MAP) -> list:
    """Flat list of all raw Product values we want to keep."""
    return [p for plist in product_map.values() for p in plist]