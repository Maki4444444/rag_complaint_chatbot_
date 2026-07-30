"""
src/generator.py

Generator module for Task 3: combines the prompt template, retrieved
chunks, and user question, then calls the HuggingFace Inference API
to produce a grounded, evidence-backed answer.

Uses Qwen/Qwen2.5-7B-Instruct via HuggingFace's free Inference API --
no model download, no GPU required. The model runs on HF servers.

Setup:
    1. Create a free account at huggingface.co
    2. Go to Settings -> Access Tokens -> New Token (Read access)
    3. Add HF_TOKEN=hf_... to your .env file in the repo root

Robustness:
    - build_llm_client() sets a request timeout so a hung connection can
      never freeze the UI indefinitely.
    - generate_answer() retries transient failures (timeouts, rate limits,
      connection errors) with exponential backoff, then returns a clear
      user-facing fallback answer instead of raising -- callers can check
      result["error"] to see whether the fallback path was taken.
    - Empty retrieved_chunks short-circuits before ever calling the LLM,
      returning a "no relevant complaints found" answer directly.
"""
import logging
import os
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()

logger = logging.getLogger(__name__)

# ── Model configuration ────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL    = "Qwen/Qwen2.5-7B-Instruct"

# ── Resilience configuration ───────────────────────────────────────────────
REQUEST_TIMEOUT_SECONDS = 20    # per-attempt network timeout
MAX_RETRIES             = 3     # total attempts on transient failure
BACKOFF_BASE_SECONDS    = 1.5   # 1.5s, 3s, 6s ...

NO_CONTEXT_MESSAGE = (
    "I couldn't find any relevant complaints for that question. "
    "Try rephrasing it, choosing a different product category, or "
    "asking about one of the four supported products (Credit Card, "
    "Personal Loan, Savings Account, Money Transfer)."
)

LLM_FAILURE_MESSAGE = (
    "I'm having trouble reaching the answer-generation service right now. "
    "Please try again in a moment. If this keeps happening, the retrieved "
    "sources below are still a good starting point."
)

# ── Prompt template ────────────────────────────────────────────────────────
# This template is a core rubric requirement:
#   - Sets the analyst role and domain (CrediTrust / financial complaints)
#   - Enforces groundedness: answer ONLY from the provided context
#   - Includes a fallback: admit when context is insufficient
RAG_PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. \
Your task is to answer questions about customer complaints.
Use ONLY the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain enough information to answer the question, \
state clearly: "I don't have enough information to answer that."
Do not add any information not present in the excerpts below.

Retrieved complaint excerpts:
{context}

Question: {question}

Answer:"""


def build_context_block(retrieved_chunks: list) -> str:
    """
    Format the list of retrieved chunks into a numbered context block
    for injection into the prompt, including key metadata for traceability.
    """
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(
            f"[Excerpt {i} | Product: {chunk['product_category']} | "
            f"Complaint ID: {chunk['complaint_id']}]\n{chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


def build_llm_client(token: str = HF_TOKEN, model: str = MODEL) -> InferenceClient:
    """
    Connect to the HuggingFace Inference API.
    Raises a clear error if the token is missing.

    A request timeout is always set so a stalled connection can never
    hang the UI indefinitely -- it will surface as a retryable failure
    instead.
    """
    if not token:
        raise ValueError(
            "HF_TOKEN is not set. Add HF_TOKEN=hf_... to your .env file.\n"
            "Get a free token at: https://huggingface.co/settings/tokens"
        )
    return InferenceClient(model=model, token=token, timeout=REQUEST_TIMEOUT_SECONDS)


def is_transient_error(exc: Exception) -> bool:
    """
    Heuristic for whether an exception from the Inference API is worth
    retrying (timeouts, rate limits, 5xx) versus failing fast (e.g. a
    401 from a bad token, which retrying will never fix).
    """
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, HfHubHTTPError):
        status = getattr(exc.response, "status_code", None)
        # 429 (rate limited) and 5xx (server-side) are worth retrying;
        # 4xx auth/validation errors are not.
        return status == 429 or (status is not None and status >= 500)
    # Fall back to retrying on generic connection-level errors, but not
    # on programming errors (KeyError, TypeError, etc.).
    return isinstance(exc, (ConnectionError, OSError))


def _call_llm_with_retry(client: InferenceClient, prompt: str, max_new_tokens: int):
    """
    Calls the LLM with exponential-backoff retry on transient failures.
    Returns the raw response object on success.
    Raises the last exception if all attempts are exhausted or the
    failure is deemed non-transient.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 -- intentionally broad; classified below
            last_exc = exc
            if not is_transient_error(exc) or attempt == MAX_RETRIES:
                logger.warning(
                    "LLM call failed on attempt %d/%d (non-retryable or exhausted): %s",
                    attempt, MAX_RETRIES, exc,
                )
                raise
            wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "LLM call failed on attempt %d/%d (%s); retrying in %.1fs",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    # Unreachable, but keeps type-checkers happy.
    raise last_exc


def generate_answer(
    client: InferenceClient,
    question: str,
    retrieved_chunks: list,
    max_new_tokens: int = 300,
) -> dict:
    """
    Full RAG generator step:
      1. Short-circuit with a clear message if no chunks were retrieved.
      2. Build a numbered context block from the retrieved chunks.
      3. Inject context + question into the prompt template.
      4. Call the LLM via HuggingFace Inference API, retrying transient
         failures with backoff.
      5. On unrecoverable failure, return a graceful fallback answer
         instead of raising.

    Args:
        client:           HuggingFace InferenceClient
        question:         the user's plain-English question
        retrieved_chunks: list of dicts from the retriever
        max_new_tokens:   max length of the generated answer

    Returns:
        dict with keys: question, answer, context_block, sources, prompt,
        error (bool -- True if this is a fallback answer, not a real
        LLM-generated one).
    """
    # ── Empty retrieval short-circuit ───────────────────────────────────
    if not retrieved_chunks:
        return {
            "question":      question,
            "answer":        NO_CONTEXT_MESSAGE,
            "context_block": "",
            "sources":       [],
            "prompt":        None,
            "error":         False,  # not a failure -- a legitimate empty result
        }

    context_block = build_context_block(retrieved_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_block,
        question=question,
    )

    # ── Generation with retry ───────────────────────────────────────────
    try:
        response = _call_llm_with_retry(client, prompt, max_new_tokens)
        answer = (response.choices[0].message.content or "").strip()
        error = False
    except Exception as exc:  # noqa: BLE001 -- final safety net, never crash the app
        logger.error("LLM generation failed after retries: %s", exc)
        answer = LLM_FAILURE_MESSAGE
        error = True

    return {
        "question":      question,
        "answer":        answer,
        "context_block": context_block,
        "sources":       retrieved_chunks,
        "prompt":        prompt,
        "error":         error,
    }