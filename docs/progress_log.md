# Progress Log

Chronological record of what was built, why, and how it was verified.
Companion to `docs/WHY_A_CLEAN_REBUILD.md` (the "why a new repo" doc) and
`docs/IMPLEMENTATION_PLAN.md` (the phased build plan). This log is the
"what actually happened" record which gets updated as work progresses, not
rewriting history in it.


## Phase 0: Repo setup

Skeleton created per `docs/IMPLEMENTATION_PLAN.md`'s structure. Carried
over unmodified from the old repo: `.github/workflows/unittests.yml`
(CI), `notebooks/rag_pipeline.ipynb` (Task 3's qualitative eval, to be
re-run once Task 3 is built).


## Phase 1 `feat/preprocessing` (Task 1)

**What was built:**
- `src/preprocessing.py` full cleaning pipeline (lowercase, boilerplate
  stripping, noise removal, NLP normalization). New vs. v1:
  `strip_boilerplate_openers()`, closing the gap where v1 never
  implemented the Task 1 brief's named boilerplate-stripping example.
- `tests/test_preprocessing.py` 26 tests (12 carried over, 14 new for
  boilerplate stripping).
- `notebooks/01_eda_preprocessing.ipynb` EDA, filtering, cleaning,
  saves `data/processed/filtered_complaints.csv`.

**Bugs found and fixed during verification (not caught by writing the
code caught by actually running it):**
- The first version of the boilerplate regex patterns used a greedy
  `[^.!]*[.!]?` tail intended to consume "the rest of the sentence."
  When there was no punctuation between the opener and the end of the
  narrative (a common case "i would like to file a complaint about X"
  often has no comma before X), the greedy match consumed the *entire*
  narrative, silently deleting real content. Fixed by matching only the
  fixed opener phrase (+ at most one immediate connector word like
  "about"/"regarding"), never an open-ended run to the next punctuation.
- The notebook's demonstration cell assumed at least one complaint in
  the sample would produce multiple chunks; a `next()` call without a
  default raised `StopIteration` when that assumption happened not to
  hold. Guarded with `next(..., None)` + a fallback message.

**Verification:**
- `pytest tests/test_preprocessing.py -v` 26/26 passing.
- Notebook executed end-to-end (`jupyter nbconvert --execute`) against a
  synthetic CFPB-shaped dataset with known boilerplate openers, mixed
  target/non-target product labels, and empty narratives 0 errors, 0
  warnings after fixes. Outputs cleared before commit.


## Phase 2 `feat/sample-vector-store` (Task 2)

> **Status: Complete.** Code written, tested against synthetic data
> (30 tests), and run end-to-end against the real dataset see
> "Update real run completed" below.

**What was built:**
- `src/sampling.py` *(new)* proportional stratified sampling, closing
  the gap where v1's 12,500-complaint sample was reported in the README
  but never had committed, reproducible sampling code.
- `src/build_sample_index.py` *(new)* the actual Task 2 deliverable
  ("a script performing chunking, embedding, and indexing"). Orchestrates
  sampling → chunking → embedding → ChromaDB indexing, writes
  `build_manifest.json` recording the exact sampled complaint IDs.
- `src/chunking.py`, `src/embedding.py` carried over from v1 unchanged
  (no functional gap), docstrings expanded with "why" justification for
  chunk_size/overlap, model choice, ChromaDB over FAISS.
- `tests/test_chunking.py`, `tests/test_embedding.py`,
  `tests/test_sampling.py`, `tests/test_build_sample_index.py` *(all
  new)* 30 tests total. v1 had zero tests for chunking/embedding.
- `notebooks/02_chunking_embedding_indexing.ipynb` visualization/
  showcase companion to the build script (not the pipeline itself):
  sample-vs-full category proportion chart, chunk_size/overlap
  experimentation (250/25 vs 500/50 vs 1000/100) with justification for
  the final 500/50 choice, a 2D PCA projection of chunk embeddings by
  category, and a live retrieval showcase query.

**Bugs found and fixed during verification:**
- `stratified_sample()`'s first implementation used
  `groupby(...).apply(lambda g: g.sample(...), include_groups=True)`.
  Recent pandas versions raise `ValueError: include_groups=True is no
  longer allowed`. Fixed by switching to pandas' native
  `GroupBy.sample(frac=..., random_state=...)`, which doesn't have this
  problem and is simpler besides.
- The notebook's embedding-visualization cell hit the same
  `groupby().apply()` issue in a different form (a `KeyError:
  'product_category'` once the `include_groups` param was simply
  omitted rather than fixed) — replaced with a plain per-category loop,
  sidestepping pandas' `groupby().apply()` grouping-column handling
  entirely rather than chasing version-specific behavior.
- A hand-constructed notebook cell (built as a raw dict during testing,
  not via `nbformat.v4.new_code_cell()`) was missing line terminators
  between source lines, silently producing malformed multi-statement
  code. Not a bug in the shipped notebook — a reminder that hand-rolled
  nbformat JSON needs the same care as any other generated artifact.
- `src/embedding.py` never loaded `.env` or authenticated with HF Hub,
  so every model download hit the unauthenticated rate limit (visible
  as `Warning: You are sending unauthenticated requests to the HF Hub`
  and slow downloads in practice). Fixed by loading `HF_TOKEN` via
  `python-dotenv`, same pattern as `src/generator.py`, passed to
  `SentenceTransformer(model_name, token=HF_TOKEN)`. Falls back to
  unauthenticated automatically if unset.

**Verification:**
- `pytest tests/test_chunking.py tests/test_embedding.py
  tests/test_sampling.py tests/test_build_sample_index.py -v` 30/30
  passing, including a full integration test that runs sample → chunk →
  embed → index against synthetic data and queries the resulting
  ChromaDB collection back. Only the embedding model itself is mocked in
  tests (no network dependency in CI); `build_sample_index.py` uses the
  real model when run for real.
- `notebooks/02_chunking_embedding_indexing.ipynb` executed end-to-end
  against synthetic data (embedding model mocked for the test run only,
  not in the shipped notebook) 0 errors across all 14 code cells after
  fixes above.
- Re-ran the affected test suites after the HF_TOKEN fix still 13/13
  passing (`test_embedding.py` + `test_build_sample_index.py`).

**Not yet done at time of writing (updated below):** the notebook and
`build_sample_index.py` had not yet been run against the *real*
480,564-row `data/processed/filtered_complaints.csv` in the sandboxed
verification environment (no network access to `huggingface.co` or the
CFPB dataset host from there)  only against synthetic data.

**Update real run completed:** `notebooks/02_chunking_embedding_indexing.ipynb`
has now been run end-to-end locally against the real
`data/processed/filtered_complaints.csv`, with real network access to
download `all-MiniLM-L6-v2` (using the `HF_TOKEN` setup from above) and
real embedding computation. `vector_store_sample/` now exists as a real,
populated ChromaDB collection built from actual CFPB complaint data, not
synthetic test fixtures. Task 2 is complete end to end: code written,
tested against synthetic data (30 tests), *and* run for real against the
target dataset.


## Phase 3 Tasks 3 & 4 (in progress)

**Status:** blocked on confirming `complaint_embeddings.parquet`'s exact
column schema (needed to write `src/build_full_index.py`, the ingestion
script for `vector_store_full/`). Runtime resilience
(`retriever.py`/`generator.py`/`app.py` error handling, retry/backoff,
graceful fallbacks) and the evaluation harness (`src/evaluation.py`,
LLM-as-judge, `--resume`, swappable models) already exist and are tested
from earlier work in this project being carried into this repo's
Task 3/4 branches rather than re-derived, per `docs/WHY_A_CLEAN_REBUILD.md`
point 4.