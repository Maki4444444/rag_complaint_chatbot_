"""
src/build_full_index.py

Task 3 deliverable step 1: "Load the pre-built vector store from the
dataset resources (covers the complete filtered dataset)."

The provided complaint_embeddings.parquet already contains precomputed
embeddings (sentence-transformers/all-MiniLM-L6-v2, 384-dim) for ~1.37M
chunks across ~464K complaints -- this script does NOT re-embed anything,
it ingests the existing embeddings into a queryable ChromaDB collection
at vector_store_full/.

Confirmed schema (via src/explore_parquet.py against the real file):
    id                string   "{complaint_id}_{chunk_index}"
    document          string   chunk text
    embedding         list<double>, len 384
    metadata          struct:
        chunk_index       int64
        company           string
        complaint_id      string
        date_received     string
        issue             string
        product           string
        product_category  string   (already "Credit Card"/"Personal Loan"/
                                     "Savings Account"/"Money Transfer" --
                                     matches src/preprocessing.py's target
                                     categories directly, no remapping
                                     needed)
        state             string
        sub_issue         string
        total_chunks      int64

Usage:
    python -m src.build_full_index --input data/raw/complaint_embeddings.parquet

--------------------------------------------------------------------------
IMPROVEMENT OVER v1 (Maki4444444/rag-complaint-chatbot):
This script did not exist in v1 -- app.py was wired to the Task 2 sample
store instead of a full-scale index, because no code ever built one. This
is the fix for the retrieval-scope gap documented as point 1 in
docs/WHY_A_CLEAN_REBUILD.md. Output goes to vector_store_full/, kept
explicitly distinct from vector_store_sample/ (Task 2's exercise) so the
two can't be confused the way they were before.

Note: unlike src/build_sample_index.py, this script is NOT built on top
of src/chunking.py/src/embedding.py's embed_chunks() -- there is no
embedding step here, since the embeddings are already computed and
provided. Reusing embed_chunks() would mean re-embedding 1.37M chunks for
no reason (slow, and would drift from the original embeddings if the
model version differs even slightly). This script only reads and writes.
--------------------------------------------------------------------------
"""
import argparse
import json
import time
from pathlib import Path

import chromadb
import pyarrow.parquet as pq

DEFAULT_INPUT_PATH = "data/raw/complaint_embeddings.parquet"
DEFAULT_OUTPUT_DIR = "vector_store_full"
DEFAULT_COLLECTION_NAME = "complaints_full"

# Row-group-sized reads keep peak memory bounded: the real file has 2 row
# groups covering 1.37M rows, so pyarrow's own batch iterator (rather
# than pandas.read_parquet loading all 2.2GB at once) is the difference
# between this running on a laptop and needing a much bigger machine.
DEFAULT_READ_BATCH_SIZE = 10_000

# ChromaDB has a per-call max batch add size; insert in smaller batches
# regardless of how big the read batches are.
CHROMA_ADD_BATCH_SIZE = 5_000

# Metadata fields expected in the `metadata` struct column, and their
# target Python types once flattened -- ChromaDB metadata values must be
# str/int/float/bool, so struct-typed nested values are not acceptable
# as-is even though the struct decodes fine in pandas.
_METADATA_INT_FIELDS = {"chunk_index", "total_chunks"}


def _flatten_metadata(meta: dict) -> dict:
    """
    Converts a row's metadata struct (already a Python dict once pyarrow
    decodes it) into a ChromaDB-safe flat dict: known int fields coerced
    to int, everything else coerced to str, missing/None values become
    empty string rather than raising -- a schema field being briefly
    absent from a future export shouldn't crash the whole ingestion run.
    """
    flat = {}
    for key, value in meta.items():
        if key in _METADATA_INT_FIELDS:
            flat[key] = int(value) if value is not None else 0
        else:
            flat[key] = str(value) if value is not None else ""
    return flat


def build_full_index(
    input_path: str = DEFAULT_INPUT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    read_batch_size: int = DEFAULT_READ_BATCH_SIZE,
    limit_rows: int | None = None,
) -> dict:
    """
    Streams complaint_embeddings.parquet in batches and ingests every row
    into a persisted ChromaDB collection. Returns a summary dict (also
    written to <output_dir>/build_manifest.json).

    limit_rows: if set, stops after ingesting this many rows -- useful
    for a fast smoke-test run before committing to the full ~1.37M-row,
    multi-hour(ish) ingestion.
    """
    t0 = time.time()

    print(f"Opening {input_path} ...")
    parquet_file = pq.ParquetFile(input_path)
    total_rows_in_file = parquet_file.metadata.num_rows
    print(f"  {total_rows_in_file:,} rows, {parquet_file.metadata.num_row_groups} row groups")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=output_dir)

    # Drop existing collection of the same name so re-runs are idempotent.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)

    rows_ingested = 0
    category_counts: dict = {}

    for batch in parquet_file.iter_batches(batch_size=read_batch_size):
        df = batch.to_pandas()

        if limit_rows is not None and rows_ingested >= limit_rows:
            break
        if limit_rows is not None:
            remaining = limit_rows - rows_ingested
            if len(df) > remaining:
                df = df.iloc[:remaining]

        ids = df["id"].astype(str).tolist()
        documents = df["document"].astype(str).tolist()
        embeddings = df["embedding"].apply(list).tolist()
        metadatas = [_flatten_metadata(m) for m in df["metadata"].tolist()]

        for m in metadatas:
            cat = m.get("product_category", "")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Insert in CHROMA_ADD_BATCH_SIZE-sized sub-batches even though we
        # already read in read_batch_size chunks, since the two batch
        # sizes serve different purposes (memory-bounded reads vs.
        # ChromaDB's own per-call add limit) and needn't be equal.
        for i in range(0, len(ids), CHROMA_ADD_BATCH_SIZE):
            collection.add(
                ids=ids[i:i + CHROMA_ADD_BATCH_SIZE],
                embeddings=embeddings[i:i + CHROMA_ADD_BATCH_SIZE],
                documents=documents[i:i + CHROMA_ADD_BATCH_SIZE],
                metadatas=metadatas[i:i + CHROMA_ADD_BATCH_SIZE],
            )

        rows_ingested += len(ids)
        elapsed = time.time() - t0
        rate = rows_ingested / elapsed if elapsed > 0 else 0
        print(f"  {rows_ingested:,}/{total_rows_in_file:,} rows ingested "
              f"({rate:.0f} rows/s, {elapsed:.0f}s elapsed)")

        if limit_rows is not None and rows_ingested >= limit_rows:
            break

    elapsed = time.time() - t0
    final_count = collection.count()

    manifest = {
        "input_path": input_path,
        "output_dir": output_dir,
        "collection_name": collection_name,
        "rows_in_source_file": total_rows_in_file,
        "rows_ingested": rows_ingested,
        "final_collection_count": final_count,
        "category_counts": category_counts,
        "limit_rows": limit_rows,
        "elapsed_seconds": round(elapsed, 1),
    }

    manifest_path = Path(output_dir) / "build_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {final_count:,} chunks indexed in {elapsed:.0f}s.")
    print(f"Manifest written to {manifest_path}")

    if final_count != rows_ingested:
        print(f"WARNING: collection.count()={final_count:,} does not match "
              f"rows_ingested={rows_ingested:,} -- check for duplicate ids "
              f"in the source file.")

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--read-batch-size", type=int, default=DEFAULT_READ_BATCH_SIZE)
    parser.add_argument("--limit-rows", type=int, default=None,
                         help="Ingest only the first N rows -- for a fast smoke test "
                              "before running the full ~1.37M-row ingestion.")
    args = parser.parse_args()

    build_full_index(
        input_path=args.input,
        output_dir=args.output,
        collection_name=args.collection,
        read_batch_size=args.read_batch_size,
        limit_rows=args.limit_rows,
    )


if __name__ == "__main__":
    main()