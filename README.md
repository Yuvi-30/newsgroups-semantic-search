# Newsgroups Semantic Search

A lightweight semantic search system over the 20 Newsgroups corpus (~19k documents), featuring fuzzy clustering and a from-scratch cluster-aware semantic cache.

---

## Architecture

```
User Query
    │
    ▼
FastAPI (api/main.py)
    │
    ▼
SearchService (src/search.py)
    │
    ├─── SemanticCache (src/cache.py)  ◄── cluster-partitioned cosine lookup
    │         │ hit                           from-scratch, no Redis
    │         ▼
    │    Return cached result
    │
    │ miss
    ▼
VectorStore (src/vector_store.py)    ◄── ChromaDB + all-MiniLM-L6-v2
    │
    ▼
Format result → Store in cache → Return
```

---

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ingest corpus (~5–10 min on CPU)

Downloads the 20 Newsgroups dataset, cleans it, embeds ~19k documents with `all-MiniLM-L6-v2`, and persists them in ChromaDB.

```bash
python -m scripts.ingest
```

### 3. Run fuzzy clustering (~3–8 min on CPU)

Runs Fuzzy C-Means (k=15, m=2.0) over the embedding matrix, saves the clustering artifact, and writes per-document cluster memberships back to ChromaDB.

```bash
python -m scripts.cluster
```

### 4. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

---

## API Endpoints

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best graphics cards for gaming?"}'
```

Response:
```json
{
  "query": "What are the best graphics cards for gaming?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "Top match (similarity: 0.812)...",
  "dominant_cluster": 3
}
```

Second request with paraphrase:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "top GPUs for PC gaming"}'
```

```json
{
  "cache_hit": true,
  "matched_query": "What are the best graphics cards for gaming?",
  "similarity_score": 0.891,
  "dominant_cluster": 3
}
```

### `GET /cache/stats`

```bash
curl http://localhost:8000/cache/stats
```

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

```bash
curl -X DELETE http://localhost:8000/cache
```

### `POST /cache/threshold` (bonus)

Adjust similarity threshold at runtime:

```bash
curl -X POST http://localhost:8000/cache/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.90}'
```

---

## Design Decisions

### Embedding Model: `all-MiniLM-L6-v2`

384-dimensional vectors. Trained on 1B+ sentence pairs. Fast enough to embed the full corpus in ~5 minutes on CPU. The symmetric training (query and document in the same space) is critical — cosine similarity between a query and a document is meaningful because they were trained together.

### Vector DB: ChromaDB

Local persistent storage. No server process. Native metadata filtering lets us pre-filter by cluster before running cosine search. The alternative (Weaviate, Qdrant) adds operational overhead that isn't warranted for a 20k-document corpus.

### Fuzzy Clustering: Fuzzy C-Means, k=15, m=2.0

The 20 newsgroup categories are not semantically orthogonal — `comp.sys.ibm.pc.hardware` and `comp.sys.mac.hardware` cluster together; `talk.politics.misc` overlaps `talk.politics.guns`. FCM gives each document a *membership distribution* across clusters, which is both more accurate and more useful downstream (the cache uses membership vectors to partition its search space).

k=15 (not 20): chosen by sweeping k=8..21 and inspecting the fuzzy partition coefficient + inertia elbow. The real semantic structure has fewer, fuzzier groups than the label taxonomy suggests.

### Semantic Cache

**Data structure:** `dict[cluster_id → list[CacheEntry]]`

**Lookup:** embed query → predict cluster membership → cosine-compare against entries in the dominant cluster bucket only (+ 2nd cluster for boundary queries).

**Complexity:** O(N/k) per lookup vs O(N) for a flat cache. With k=15 and balanced buckets, this is ~15x faster at scale.

**The threshold (θ):**

| θ | Behaviour |
|---|---|
| 0.70 | Catches paraphrases and synonym swaps. Risk of false positives on related-but-different queries. |
| 0.85 | **Default.** Catches near-identical phrasings reliably. Misses loose paraphrases. |
| 0.95 | Near-exact match only. Very low hit rate; cache barely helps. |

The interesting insight: at θ=0.85, your hit log is essentially a dictionary of what the embedding model considers synonymous. At θ=0.70, you start seeing the model's failure modes.

---

## Docker

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t semantic-search .
docker run -p 8000:8000 \
  -v $(pwd)/embeddings:/app/embeddings \
  -v $(pwd)/models:/app/models \
  semantic-search
```

**Note:** Run `python -m scripts.ingest` and `python -m scripts.cluster` before building the Docker image so the `embeddings/` and `models/` directories exist.

---

## Project Structure

```
newsgroups-semantic-search/
├── api/
│   └── main.py              # FastAPI app + all endpoints
├── src/
│   ├── preprocessing.py     # Corpus cleaning pipeline
│   ├── vector_store.py      # ChromaDB wrapper + embedding
│   ├── clustering.py        # Fuzzy C-Means + analysis utilities
│   ├── cache.py             # From-scratch semantic cache
│   └── search.py            # Orchestration layer
├── scripts/
│   ├── ingest.py            # One-time: clean + embed + store
│   └── cluster.py           # One-time: cluster + update metadata
├── notebooks/
│   └── cluster_analysis.ipynb
├── embeddings/              # ChromaDB (created at ingest time)
├── models/                  # Clustering artifact (created at cluster time)
├── .env                     # Configuration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```