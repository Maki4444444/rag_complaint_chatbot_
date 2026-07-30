"""
explore_parquet.py

Memory-safe exploration of the pre-built, embedded complaint parquet
(~1.37M rows: id, document, embedding, metadata) BEFORE writing
src/build_full_index.py against it.

Why streaming, not pd.read_parquet(path): the embedding column alone is
~1.37M rows x 384 dims x 4 bytes (float32) ~= 2.1 GB, or ~4.2 GB at
float64 -- on top of the document text and metadata dicts. Loading that
in one shot is a plausible cause of a previous crash on a resource-
constrained machine. Every full-dataset pass below reads in batches via
pyarrow and, where possible, skips the embedding column entirely.

Usage:
    python explore_parquet.py /path/to/complaint_embeddings.parquet
    python explore_parquet.py /path/to/complaint_embeddings.parquet --full-scan
    python explore_parquet.py /path/to/complaint_embeddings.parquet --batch-size 2000
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

DEFAULT_DATA_DIR = "data/raw"


def resolve_default_path(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    No explicit path given -- look in data/raw/ for a .parquet file.
    Errors out with a clear message rather than guessing wrong silently.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"No path given and '{data_dir}/' doesn't exist. "
            f"Pass the path explicitly: python explore_parquet.py /path/to/file.parquet"
        )
    candidates = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not candidates:
        raise FileNotFoundError(
            f"No .parquet file found in '{data_dir}/'. "
            f"Pass the path explicitly: python explore_parquet.py /path/to/file.parquet"
        )
    if len(candidates) > 1:
        print(f"Multiple .parquet files found in '{data_dir}/', using the first: "
              f"{candidates[0]}")
        print(f"  (others: {candidates[1:]})")
    return candidates[0]


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def inspect_schema(path: str) -> pq.ParquetFile:
    """Reads only file-level metadata -- no row data touched."""
    pf = pq.ParquetFile(path)
    file_size = os.path.getsize(path)

    print("=" * 70)
    print("SCHEMA & FILE-LEVEL METADATA")
    print("=" * 70)
    print(f"File path:        {path}")
    print(f"File size on disk: {human_bytes(file_size)}")
    print(f"Total rows:        {pf.metadata.num_rows:,}")
    print(f"Row groups:        {pf.metadata.num_row_groups}")
    print(f"Columns:           {pf.schema.names}")
    print()
    print("Arrow schema:")
    print(pf.schema_arrow)
    print()
    return pf


def sample_rows(pf: pq.ParquetFile, n: int = 5):
    """Materializes only the first ~n rows (one small batch), full columns
    included -- safe regardless of total file size."""
    print("=" * 70)
    print(f"SAMPLE ROWS (first {n})")
    print("=" * 70)
    batch = next(pf.iter_batches(batch_size=n))
    df = batch.to_pandas()

    for i, row in df.head(n).iterrows():
        print(f"\n--- row {i} ---")
        print(f"id:        {row['id']}")
        print(f"document:  {row['document'][:200]}"
              f"{'...' if len(row['document']) > 200 else ''}")
        emb = row["embedding"]
        print(f"embedding: len={len(emb)}, dtype={type(emb[0]).__name__}, "
              f"first 5 values={list(emb[:5])}")
        print(f"metadata keys: {sorted(row['metadata'].keys())}")
        print(f"metadata:  {json.dumps(row['metadata'], indent=2, default=str)}")
    print()
    return df


def scan_metadata_keys(pf: pq.ParquetFile, n_batches: int = 20, batch_size: int = 2000):
    """
    Scans several batches (not the whole file, unless n_batches is large)
    to find:
      - the full union of metadata keys actually seen (schemas can be
        inconsistent row to row in a dict-typed column)
      - which keys are present in every row sampled vs. only sometimes
      - the set of distinct value-types seen per key
    """
    print("=" * 70)
    print(f"METADATA KEY SCAN (up to {n_batches} batches x {batch_size} rows)")
    print("=" * 70)

    key_counts = Counter()
    value_type_by_key = defaultdict(set)
    rows_seen = 0

    reader = pf.iter_batches(batch_size=batch_size, columns=["metadata"])
    for i, batch in enumerate(reader):
        if i >= n_batches:
            break
        df = batch.to_pandas()
        rows_seen += len(df)
        for meta in df["metadata"]:
            for k, v in meta.items():
                key_counts[k] += 1
                value_type_by_key[k].add(type(v).__name__)

    print(f"Rows sampled: {rows_seen:,}\n")
    print(f"{'key':<25} {'present in':>12} {'coverage':>10}   value types")
    print("-" * 70)
    for key, count in key_counts.most_common():
        coverage = f"{count / rows_seen:.1%}"
        types = ", ".join(sorted(value_type_by_key[key]))
        print(f"{key:<25} {count:>10,}/{rows_seen:<6} {coverage:>10}   {types}")
    print()
    return key_counts, value_type_by_key


def id_pattern_check(pf: pq.ParquetFile, n_batches: int = 5, batch_size: int = 5000):
    """
    Checks whether `id` really follows `{complaint_id}_{chunk_index}` and
    whether that chunk_index matches metadata['chunk_index'] (if present),
    on a sample -- informs whether we can derive complaint_id/chunk_index
    from `id` alone if metadata lacks them.
    """
    print("=" * 70)
    print("ID PATTERN CHECK")
    print("=" * 70)

    mismatches = 0
    checked = 0
    reader = pf.iter_batches(batch_size=batch_size, columns=["id", "metadata"])
    for i, batch in enumerate(reader):
        if i >= n_batches:
            break
        df = batch.to_pandas()
        for _id, meta in zip(df["id"], df["metadata"]):
            checked += 1
            parts = _id.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                print(f"  Unexpected id format: {_id!r}")
                continue
            complaint_id, idx_from_id = parts[0], int(parts[1])
            idx_from_meta = meta.get("chunk_index")
            if idx_from_meta is not None and idx_from_meta != idx_from_id:
                mismatches += 1
                if mismatches <= 5:
                    print(f"  Mismatch: id={_id!r} -> chunk_index={idx_from_id}, "
                          f"but metadata['chunk_index']={idx_from_meta}")

    print(f"\nChecked {checked:,} ids. Mismatches: {mismatches}")
    print()


def full_scan_stats(pf: pq.ParquetFile, batch_size: int = 5000):
    """
    One full pass over the dataset, embedding column excluded, to compute:
      - total row count (cross-check against file metadata)
      - distinct complaint count (derived from id prefix) and chunks/complaint stats
      - document (chunk text) length distribution
      - value counts for likely category-ish metadata fields, if present
    This still touches every row, so it's the slowest option -- use
    --full-scan to opt in.
    """
    print("=" * 70)
    print("FULL DATASET SCAN (embedding column excluded)")
    print("=" * 70)

    total_rows = 0
    complaint_chunk_counts = Counter()
    doc_lengths = []
    category_like_counters = defaultdict(Counter)
    # Field names worth tallying if present -- extend once real keys are known.
    CATEGORY_CANDIDATE_KEYS = {"product", "product_category", "Product", "category"}

    columns = ["id", "document", "metadata"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        total_rows += len(df)

        for _id in df["id"]:
            complaint_id = _id.rsplit("_", 1)[0]
            complaint_chunk_counts[complaint_id] += 1

        doc_lengths.extend(df["document"].str.len().tolist())

        for meta in df["metadata"]:
            for k in CATEGORY_CANDIDATE_KEYS & meta.keys():
                category_like_counters[k][meta[k]] += 1

        if total_rows % (batch_size * 50) == 0:
            print(f"  ...scanned {total_rows:,} rows")

    print(f"\nTotal rows scanned:        {total_rows:,}")
    print(f"Distinct complaint IDs:    {len(complaint_chunk_counts):,}")
    chunks_per = list(complaint_chunk_counts.values())
    print(f"Chunks per complaint:      "
          f"min={min(chunks_per)}, max={max(chunks_per)}, "
          f"mean={sum(chunks_per)/len(chunks_per):.2f}")

    doc_lengths.sort()
    n = len(doc_lengths)
    def pct(p):
        return doc_lengths[int(n * p)]
    print(f"\nChunk (document) character length distribution:")
    print(f"  min={doc_lengths[0]}, p25={pct(0.25)}, median={pct(0.5)}, "
          f"p75={pct(0.75)}, p95={pct(0.95)}, max={doc_lengths[-1]}")

    for key, counter in category_like_counters.items():
        print(f"\nValue counts for metadata['{key}']:")
        for val, cnt in counter.most_common(20):
            print(f"  {val!r}: {cnt:,}")
    print()


def embedding_dim_check(pf: pq.ParquetFile, n: int = 500):
    print("=" * 70)
    print(f"EMBEDDING DIMENSION CHECK (first {n} rows)")
    print("=" * 70)
    batch = next(pf.iter_batches(batch_size=n, columns=["embedding"]))
    df = batch.to_pandas()
    dims = df["embedding"].apply(len)
    print(f"Distinct embedding lengths seen: {sorted(dims.unique().tolist())}")
    if dims.nunique() == 1:
        print(f"-> Consistent {dims.iloc[0]}-dim vectors "
              f"({'matches' if dims.iloc[0] == 384 else 'does NOT match'} "
              f"all-MiniLM-L6-v2's 384 dims)")
    else:
        print("-> WARNING: inconsistent embedding dimensions in this sample!")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_path", nargs="?", default=None,
                         help=f"Path to the embedded parquet file. If omitted, "
                              f"looks for a single .parquet file in "
                              f"'{DEFAULT_DATA_DIR}/'.")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--full-scan", action="store_true",
                         help="Run the full-dataset pass (slower; embedding "
                              "column still excluded). Without this flag, "
                              "only fast, sample-based checks run.")
    args = parser.parse_args()

    parquet_path = args.parquet_path or resolve_default_path()
    pf = inspect_schema(parquet_path)
    sample_rows(pf, n=5)
    scan_metadata_keys(pf, n_batches=20, batch_size=args.batch_size)
    id_pattern_check(pf, n_batches=5, batch_size=args.batch_size)
    embedding_dim_check(pf, n=500)

    if args.full_scan:
        full_scan_stats(pf, batch_size=args.batch_size)
    else:
        print("Skipped full dataset scan (pass --full-scan to run it; "
              "reads every row once, embedding column excluded, so it's "
              "safe on memory but will take a while on 1.37M rows).")


if __name__ == "__main__":
    sys.exit(main())