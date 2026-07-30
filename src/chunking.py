"""
src/chunking.py

Text chunking utilities for Task 2: splitting long complaint narratives
into smaller, embedding-friendly chunks.

--------------------------------------------------------------------------
CARRIED OVER FROM v1 (Maki4444444/rag-complaint-chatbot), UNCHANGED:
No functional gap was found here -- chunk_size/overlap already matched the
Tasks 3-4 pre-built store spec, and the splitter choice was already
justified. Only the "why" commentary below is new, for the report/branch
documentation. See docs/WHY_A_CLEAN_REBUILD.md for what *did* change
elsewhere in this rebuild.
--------------------------------------------------------------------------
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter(chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter.

    Why RecursiveCharacterTextSplitter over a naive fixed-width splitter:
    it tries a priority list of separators ("\\n\\n", "\\n", ". ", " ", "")
    and only falls back to a harder mid-word cut when none of those exist
    within chunk_size -- so most chunks end on a paragraph, sentence, or
    at worst word boundary rather than mid-token, which keeps each chunk
    semantically coherent for the embedding model.

    Why chunk_size=500 / chunk_overlap=50 specifically: this matches the
    pre-built complaint_embeddings.parquet vector store spec used in
    Tasks 3-4 (500 chars / 50 char overlap), so chunk granularity is
    consistent whether a query hits this Task 2 sample index or the full
    Task 3/4 index -- retrieval behavior doesn't shift depending on which
    store is being searched. 500 characters is roughly 80-100 words,
    short enough that a single retrieved chunk stays topically focused
    (one complaint's narrative is often 100+ words per the Task 1 EDA,
    so most narratives split into multiple chunks rather than one),
    while the 50-char overlap prevents a sentence that happens to fall
    exactly on a chunk boundary from being split with its context lost
    on both sides.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_dataframe(df, text_column: str, splitter: RecursiveCharacterTextSplitter, id_column: str = "Complaint ID"):
    """
    Splits every narrative in `df[text_column]` into chunks and returns a
    flat list of dicts, one per chunk, with metadata for traceability.

    Each dict contains:
        complaint_id, product_category, chunk_index, total_chunks, chunk_text

    chunk_index/total_chunks are kept explicitly (rather than only an
    opaque chunk id) so that at retrieval time a UI or evaluator can show
    "this is chunk 2 of 4 from complaint #123" -- useful both for the
    Task 4 source-display requirement and for spotting when a retrieved
    chunk is missing surrounding context from the same complaint.
    """
    records = []

    for _, row in df.iterrows():
        text = row[text_column]
        if not isinstance(text, str) or text.strip() == "":
            continue

        chunks = splitter.split_text(text)
        total_chunks = len(chunks)

        for idx, chunk_text in enumerate(chunks):
            records.append({
                "complaint_id": row[id_column],
                "product_category": row["product_category"],
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_text": chunk_text,
            })

    return records