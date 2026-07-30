"""
src/retriever.py

Retriever module for Task 3: embeds a user question and performs
semantic similarity search against the ChromaDB vector store,
returning the top-k most relevant complaint chunks with metadata.

Supports optional product_category filtering so questions about a
specific product only search within that category's chunks.

Points at vector_store_full/ (collection "complaints_full") by default --
the pre-built full-scale store covering all ~1.37M chunks / ~464K
complaints, per the Task 3 brief ("load the pre-built vector store... 
covers the complete filtered dataset"). vector_store_sample/ (Task 2's
12,500-complaint exercise) is a separate, explicitly-named directory --
see docs/WHY_A_CLEAN_REBUILD.md point 1 for why that distinction matters
(the previous submission's app.py was wired to the sample store instead,
because both stores shared one ambiguous vector_store/ name).

Robustness:
    - Rejects empty/whitespace-only or excessively long questions with a
      clear, catchable error instead of sending garbage to the embedding
      model or the vector store.
    - Wraps the ChromaDB query in a try/except so a transient store
      error surfaces as a RetrievalError the caller can handle gracefully,
      rather than an unhandled exception that crashes the app.
    - Returns an empty list (not an error) when the search legitimately
      finds nothing, so callers can show a "no relevant complaints found"
      message instead of a broken prompt.
"""
import chromadb
from src.embedding import load_embedding_model, EMBEDDING_MODEL_NAME

VALID_CATEGORIES = {
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
}

# Guardrail so a pasted-in essay or garbage input doesn't get embedded.
MAX_QUESTION_CHARS = 1000

DEFAULT_VECTOR_STORE_DIR = "vector_store_full"
DEFAULT_COLLECTION_NAME = "complaints_full"


class RetrievalError(Exception):
    """Raised when the vector store query itself fails (e.g. DB/connection
    issue), as opposed to a legitimate zero-result search."""


def load_vector_store(persist_dir: str = DEFAULT_VECTOR_STORE_DIR, collection_name: str = DEFAULT_COLLECTION_NAME):
    """
    Load the persisted ChromaDB collection from disk.

    Defaults to vector_store_full/ (built by src/build_full_index.py from
    the provided complaint_embeddings.parquet -- ~1.37M chunks, ~464K
    complaints), matching the Task 3 brief. Pass persist_dir=
    "vector_store_sample", collection_name="complaints_sample" explicitly
    for local dev/testing against the much smaller Task 2 index instead.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)
    print(f"Loaded collection '{collection_name}' with {collection.count():,} chunks")
    return collection


def validate_question(question: str) -> str:
    """
    Validate and normalize a raw user question.

    Raises:
        ValueError: if the question is empty/whitespace-only, not a
                    string, or exceeds MAX_QUESTION_CHARS.

    Returns:
        the stripped question string.
    """
    if not isinstance(question, str):
        raise ValueError("Question must be text.")

    stripped = question.strip()
    if not stripped:
        raise ValueError("Question cannot be empty. Please type a question.")

    if len(stripped) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"Question is too long ({len(stripped)} characters). "
            f"Please limit questions to {MAX_QUESTION_CHARS} characters."
        )

    return stripped


def build_retriever(collection, model=None):
    """
    Returns a retriever function bound to the given ChromaDB collection
    and embedding model.

    The returned retrieve() function accepts:
        question         -- plain-English question string
        top_k             -- number of chunks to return (default 5)
        product_category -- optional category filter, one of:
                            "Credit Card", "Personal Loan",
                            "Savings Account", "Money Transfer"
                            If None, searches across all categories.
    """
    if model is None:
        model = load_embedding_model(EMBEDDING_MODEL_NAME)

    def retrieve(question: str, top_k: int = 5, product_category: str = None) -> list:
        """
        Embed the question and return the top_k most semantically similar
        complaint chunks from the vector store.

        Args:
            question:         plain-English question from the user
            top_k:            number of chunks to retrieve (default 5)
            product_category: optional filter — restrict search to one of:
                              "Credit Card", "Personal Loan",
                              "Savings Account", "Money Transfer"
                              If None, searches across all categories.

        Returns:
            list of dicts, each containing:
                chunk_text, complaint_id, product_category,
                chunk_index, total_chunks, similarity_score,
                company, issue, sub_issue, date_received
            The last four are present in the full store (built by
            src/build_full_index.py from complaint_embeddings.parquet)
            but not the Task 2 sample store (built by
            src/build_sample_index.py, which only carries the minimal
            fields) -- default to "" via .get() either way, so callers
            can display them when available without needing to know
            which store is currently loaded.
            An empty list means the search legitimately found nothing.

        Raises:
            ValueError:     malformed/empty input or invalid category.
            RetrievalError: the vector store query itself failed.
        """
        # Validate free-text input first -- fail fast, before touching
        # the embedding model or the store.
        question = validate_question(question)

        # Validate category filter if provided
        if product_category and product_category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid product_category '{product_category}'. "
                f"Must be one of: {sorted(VALID_CATEGORIES)}"
            )

        # Embed the question using the same model that built the index
        try:
            query_embedding = (
                model.encode(question, normalize_embeddings=True)
                .tolist()
            )
        except Exception as exc:
            raise RetrievalError(f"Failed to embed question: {exc}") from exc

        # Build optional metadata filter for ChromaDB
        where_filter = None
        if product_category:
            where_filter = {"product_category": {"$eq": product_category}}

        # Similarity search against the vector store
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                where=where_filter,
            )
        except Exception as exc:
            raise RetrievalError(f"Vector store query failed: {exc}") from exc

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        # Legitimate "nothing found" case -- return empty list, not an error.
        if not documents or not documents[0]:
            return []

        # Flatten into a clean list of dicts
        chunks = []
        for doc, meta, dist in zip(documents[0], metadatas[0], distances[0]):
            chunks.append({
                "chunk_text": doc,
                "complaint_id": meta.get("complaint_id", ""),
                "product_category": meta.get("product_category", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                # ChromaDB returns L2 distance; convert to similarity
                "similarity_score": round(1 - dist, 4),
                # Present in the full store, absent from the Task 2
                # sample store -- default to "" rather than KeyError so
                # this function works against either.
                "company": meta.get("company", ""),
                "issue": meta.get("issue", ""),
                "sub_issue": meta.get("sub_issue", ""),
                "date_received": meta.get("date_received", ""),
            })

        return chunks

    return retrieve