"""
src/build_sample_index.py

Task 2 deliverable: "A script performing chunking, embedding, and
indexing." Ties together sampling.py, chunking.py, and embedding.py to
build the Task 2 sample vector store from data/processed/filtered_complaints.csv.

Usage:
    python -m src.build_sample_index
    python -m src.build_sample_index --sample-size 12500 --output ./vector_store_sample

--------------------------------------------------------------------------
IMPROVEMENT OVER v1 (Maki4444444/rag-complaint-chatbot):
This script did not exist in v1 -- the 12,500-complaint sample store was
built by some process that was run but never committed, so v1's
vector_store/ was reproducible in name only. Running this script is now
the single, deterministic way to (re)produce vector_store_sample/, and
it prints/saves a manifest so the exact sampled complaint IDs are
recorded, not just the resulting counts. See docs/WHY_A_CLEAN_REBUILD.md.

Note on directory naming: this writes to vector_store_sample/, not
vector_store/, to make it unambiguous that this is the Task 2 learning-
exercise index, distinct from the Tasks 3-4 full-scale index (built
separately from the provided complaint_embeddings.parquet resource into
vector_store_full/) that app.py actually serves queries against. v1's
single ambiguously-named vector_store/ directory being loaded by app.py
was the root cause of the Task 3 retrieval-scope gap documented in
docs/WHY_A_CLEAN_REBUILD.md -- two clearly-named directories make that
mistake harder to repeat by accident.
--------------------------------------------------------------------------
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.sampling import stratified_sample, summarize_sample
from src.chunking import get_text_splitter, chunk_dataframe
from src.embedding import load_embedding_model, embed_chunks, build_chroma_store, EMBEDDING_MODEL_NAME

DEFAULT_INPUT_PATH = "data/processed/filtered_complaints.csv"
DEFAULT_OUTPUT_DIR = "vector_store_sample"
DEFAULT_COLLECTION_NAME = "complaints_sample"
DEFAULT_SAMPLE_SIZE = 12_500
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def build_sample_index(
    input_path: str = DEFAULT_INPUT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    random_state: int = 42,
) -> dict:
    """
    Runs the full Task 2 pipeline and returns a summary dict (also
    written to <output_dir>/build_manifest.json for traceability).
    """
    t0 = time.time()

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  {len(df):,} rows loaded")

    print(f"\nDrawing stratified sample (target n={sample_size:,}) ...")
    sample_df = stratified_sample(df, n=sample_size, random_state=random_state)
    print(f"  {len(sample_df):,} rows sampled")
    print(summarize_sample(sample_df))

    print(f"\nChunking with chunk_size={chunk_size}, chunk_overlap={chunk_overlap} ...")
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_records = chunk_dataframe(
        sample_df, text_column="cleaned_narrative", splitter=splitter, id_column="Complaint ID"
    )
    print(f"  {len(chunk_records):,} chunks produced from {len(sample_df):,} complaints")

    print(f"\nLoading embedding model ({EMBEDDING_MODEL_NAME}) ...")
    model = load_embedding_model()

    print("Embedding chunks ...")
    chunk_records = embed_chunks(model, chunk_records)

    print(f"\nBuilding ChromaDB collection '{collection_name}' at {output_dir} ...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    collection = build_chroma_store(chunk_records, persist_dir=output_dir, collection_name=collection_name)
    print(f"  {collection.count():,} chunks indexed")

    elapsed = time.time() - t0

    manifest = {
        "input_path": input_path,
        "output_dir": output_dir,
        "collection_name": collection_name,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(sample_df),
        "num_chunks": len(chunk_records),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "random_state": random_state,
        "category_counts": summarize_sample(sample_df)["count"].to_dict(),
        # Sampled complaint IDs, not just counts -- lets anyone verify
        # exactly which rows ended up in the index, not just how many.
        "sampled_complaint_ids": sorted(sample_df["Complaint ID"].astype(str).tolist()),
        "elapsed_seconds": round(elapsed, 1),
    }

    manifest_path = Path(output_dir) / "build_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print(f"Done in {elapsed:.1f}s")

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    build_sample_index(
        input_path=args.input,
        output_dir=args.output,
        collection_name=args.collection,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()