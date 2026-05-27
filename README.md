# Semantic Search Engine

A semantic search system that understands meaning, not just keywords. Uses sentence embeddings + FAISS vector index to find conceptually similar documents.

## Architecture

```
Query Text
    ↓
Embedding Model (sentence-transformers)
    ↓
Query Vector (384-dim float32)
    ↓
FAISS Index (ANN search, cosine similarity)
    ↓
Top-K document IDs → PostgreSQL (fetch metadata)
    ↓
Ranked Results
```

## Why this beats keyword search

| Scenario | Keyword Search | Semantic Search |
|---|---|---|
| Query: "cheap flights" | Finds docs with "cheap" + "flights" | Also finds "affordable airfare", "budget travel" |
| Query: "heart attack" | Misses "myocardial infarction" | Correctly matches medical term |
| Typos | Often fails | Embedding is typo-tolerant |

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Embedding model | all-MiniLM-L6-v2 | 384-dim, 5x faster than large models, 90% quality |
| Vector index | FAISS IVF + PQ | Scales to 100M vectors; exact search only works to ~1M |
| Similarity metric | Cosine (not Euclidean) | Length-invariant — short and long docs comparable |
| Re-ranking | Cross-encoder on top-100 | ANN retrieves candidates; cross-encoder re-ranks precisely |
| Index updates | Batch rebuild nightly | FAISS doesn't support live updates; new docs go in a buffer |

## Running Locally

```bash
pip install -r requirements.txt
python -m app.main
```

## API

```
POST /index          — add documents to the index
POST /search         — semantic search query
GET  /health
```

## What to implement next

- [ ] Hybrid search: combine BM25 (keyword) + semantic scores (reciprocal rank fusion)
- [ ] Cross-encoder re-ranking for top-100 candidates
- [ ] Incremental index updates without full rebuild
- [ ] Query caching: identical queries hit Redis before FAISS

## Production Challenges

- Embedding drift requires full re-indexing when models change
- ANN introduces recall/latency tradeoffs
- Large IVF indexes require retraining to maintain cluster quality
- Hot queries benefit from Redis caching
- Distributed vector search requires shard routing and replica consistency

**"Why FAISS over exact nearest-neighbor search?"**
Exact search is O(N×D) — for 10M docs at 384 dims, that's billions of operations per query. FAISS IVF partitions the space into clusters, searching only relevant ones. 100x faster with <5% accuracy loss.

**"What is the embedding model doing?"**
It maps text to a point in 384-dimensional space such that semantically similar texts are geometrically close. Trained on millions of sentence pairs using contrastive learning.

**"How do you handle index updates when new documents arrive?"**
FAISS doesn't support live insertion into an IVF index. Two options: (1) maintain a small exact-search buffer for new docs, merge nightly; (2) use a database like Qdrant/Weaviate that supports live updates natively.
