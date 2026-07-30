"""
src/embedding.py

Embedding and vector store utilities for Task 2: turning text chunks into
vectors and persisting them in a ChromaDB collection with metadata.

--------------------------------------------------------------------------
CARRIED OVER FROM v1 (Maki4444444/rag-complaint-chatbot), UNCHANGED:
No functional gap was found here. Only the "why" commentary below is new.
--------------------------------------------------------------------------
"""
import os

import chromadb
from dotenv import load_dotenv

# Same pattern already used in src/generator.py and app.py: loads HF_TOKEN
# from .env if present. Not required for a public model like
# all-MiniLM-L6-v2, but authenticated requests get a much higher HF Hub
# rate limit and faster downloads than the default unauthenticated path
# (see the "unauthenticated requests" warning otherwise printed on load).
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None) or None  # None, not "", if unset


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Loads the sentence-transformers embedding model.

    Passes HF_TOKEN (from .env) through if set, so the model download
    uses HF Hub's authenticated rate limit rather than the default
    unauthenticated one. Falls back to the unauthenticated path
    automatically if HF_TOKEN isn't set -- this still works either way,
    just slower and with a lower request ceiling.

    Why all-MiniLM-L6-v2: it's the model named as a starting point in the
    Task 2 brief, and for this project's scale it's a reasonable choice
    on its own merits, not just because it's suggested -- 384-dimensional
    embeddings keep the ChromaDB index small (matters at the full-dataset
    scale: ~1.37M chunks x 384 floats vs. a larger model's 768+), and it
    runs fast enough on CPU that embedding a 12.5K-complaint sample here,
    or re-embedding a user's query at retrieval time in app.py, doesn't
    require a GPU. The tradeoff is somewhat lower semantic accuracy than
    a larger model (e.g. all-mpnet-base-v2) -- acceptable here since
    complaint narratives are short, topically narrow (4 product
    categories), and this is a business-facing internal tool rather than
    an open-domain search engine.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, token=HF_TOKEN)


def embed_chunks(model, chunk_records: list, batch_size: int = 64):
    """
    Generates embeddings for a list of chunk dicts (from chunk_dataframe).
    Returns the same list with an added "embedding" key per record.

    batch_size=64 balances throughput against memory: sentence-transformers
    batches internally regardless, but an explicit batch_size keeps peak
    memory bounded and predictable when this runs against the full
    Tasks 3-4 corpus (1.37M chunks) rather than only this Task 2 sample.
    """
    texts = [r["chunk_text"] for r in chunk_records]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    for record, embedding in zip(chunk_records, embeddings):
        record["embedding"] = embedding.tolist()

    return chunk_records


def build_chroma_store(chunk_records: list, persist_dir: str, collection_name: str = "complaints"):
    """
    Creates (or overwrites) a persisted ChromaDB collection from chunk
    records that already have an "embedding" key.

    Each chunk's metadata includes complaint_id, product_category,
    chunk_index, and total_chunks for traceability back to the source
    complaint -- this is what lets app.py show "source complaints"
    alongside a generated answer (Task 4's trust/usability requirement)
    rather than just a raw similarity score.

    Why ChromaDB over FAISS: ChromaDB persists metadata alongside vectors
    natively (a FAISS index is vectors-only; metadata has to be tracked
    in a parallel structure and kept in sync by hand), and its
    PersistentClient handles on-disk storage without extra plumbing --
    simpler for a project where the store is rebuilt occasionally, not
    a hot path optimized for maximum query throughput.

    Existing collection of the same name is dropped first so re-running
    this is idempotent (safe to re-run after a preprocessing fix without
    manually clearing the old index first).
    """
    client = chromadb.PersistentClient(path=persist_dir)

    # Drop existing collection of the same name so re-runs are idempotent
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    ids = [f"{r['complaint_id']}_{r['chunk_index']}" for r in chunk_records]
    embeddings = [r["embedding"] for r in chunk_records]
    documents = [r["chunk_text"] for r in chunk_records]
    metadatas = [
        {
            "complaint_id": str(r["complaint_id"]),
            "product_category": str(r["product_category"]),
            "chunk_index": int(r["chunk_index"]),
            "total_chunks": int(r["total_chunks"]),
        }
        for r in chunk_records
    ]

    # Chroma has a max batch add size; insert in batches to be safe.
    BATCH = 5000
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            documents=documents[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )

    return collection