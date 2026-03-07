"""
cache.py
--------
A semantic cache built from scratch. No Redis, no Memcached, no caching library.

DESIGN OVERVIEW
===============
Traditional caches use exact-match keys (MD5 of the query string). This breaks
the moment two users ask semantically identical questions in different words:
  "What are the best graphics cards?" vs "top GPUs for gaming?"

Our cache uses cosine similarity on query embeddings to recognise these as the
same query. The core data structure is a dict of lists, keyed by cluster ID:

    {
        cluster_id (int): [CacheEntry, CacheEntry, ...],
        ...
    }

When a new query arrives:
  1. Embed the query → embedding vector (384-dim)
  2. Predict its fuzzy membership → membership vector (15-dim)
  3. Identify the dominant cluster (argmax of membership)
  4. Cosine-compare the new query embedding against ONLY the entries in that
     cluster's bucket.
  5. If max similarity > threshold → HIT (return cached result)
  6. Else → MISS (compute result, store in the dominant cluster's bucket)

WHY CLUSTER-PARTITIONED LOOKUP:
  A naive cache compares every new query against every cached entry — O(N) per
  lookup. With 10,000 cached queries, that is 10,000 cosine comparisons per
  request.

  The cluster partitioning assumes semantically similar queries land in the same
  cluster. A query about space telescopes will not be confused with a query about
  car engines, so we never compare across those buckets. In practice, each bucket
  holds ~1/n_clusters of all entries, so lookup is O(N/k) ≈ O(N/15).

  Boundary queries (high entropy membership) could be compared against their top-2
  clusters — we implement this as an option (multi_cluster_lookup=True).

THE THRESHOLD — THE MOST IMPORTANT TUNABLE
===========================================
The similarity threshold θ controls the trade-off between cache precision and
recall. Its behaviour is non-obvious:

  θ = 0.70  (loose):
    - High recall: catches paraphrases, spelling variations, synonym swaps.
    - Risk of false positives: "nuclear power safety" and "nuclear weapons safety"
      both map to ~0.73 similarity. The cache may return a wrong result.
    - Best when: queries are repetitive, precision matters less than speed.

  θ = 0.85  (balanced, our default):
    - Catches near-identical phrasings ("best GPU" / "top graphics card") reliably.
    - Misses loose paraphrases ("what card should I buy for gaming?").
    - The empirically safe zone for English query pairs on MiniLM.

  θ = 0.95  (strict):
    - Almost no false positives — only catches essentially the same sentence
      with minor punctuation differences.
    - Very low hit rate; the cache barely helps.
    - Best when: result correctness is critical and queries are highly varied.

  The interesting insight: raising θ doesn't just reduce hits — it reveals which
  query pairs the model considers genuinely equivalent. At θ=0.85, you can read
  off the model's implicit synonym dictionary from your hit log.

  We expose threshold as a runtime-configurable parameter so callers can tune it
  per-deployment without restarting the service.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.clustering import predict_membership

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached query-result pair."""
    query: str                         # original query text
    embedding: np.ndarray              # normalised 384-dim vector
    result: str                        # the computed answer/search result
    dominant_cluster: int              # argmax of membership at cache time
    membership_vector: np.ndarray      # full soft membership (used for multi-cluster)
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0                 # how many times this entry has been a hit


@dataclass
class CacheLookupResult:
    """Outcome of a cache lookup."""
    hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: Optional[str] = None
    dominant_cluster: Optional[int] = None
    entry: Optional[CacheEntry] = None


class SemanticCache:
    """
    Cluster-partitioned semantic cache.

    Thread safety: not implemented — for a production multi-worker FastAPI
    deployment you would add a threading.Lock per bucket. Single-worker uvicorn
    (the submission requirement) is safe without it.
    """

    def __init__(
        self,
        centroids: np.ndarray,             # (n_clusters, embedding_dim) — FCM centroids
        fuzzifier: float = 2.0,
        similarity_threshold: float = 0.85,
        multi_cluster_lookup: bool = True,  # also search 2nd-best cluster for boundary docs
        top_k_clusters: int = 2,            # how many clusters to search if multi_cluster_lookup
    ):
        self.centroids = centroids
        self.n_clusters = len(centroids)
        self.fuzzifier = fuzzifier
        self.similarity_threshold = similarity_threshold
        self.multi_cluster_lookup = multi_cluster_lookup
        self.top_k_clusters = top_k_clusters

        # Core data structure: dict[cluster_id → list[CacheEntry]]
        # A defaultdict would be cleaner but a plain dict makes the structure
        # explicit — important since this is a from-scratch implementation.
        self._buckets: dict[int, list[CacheEntry]] = {
            i: [] for i in range(self.n_clusters)
        }

        # Stats
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, query: str, query_embedding: np.ndarray) -> CacheLookupResult:
        """
        Check whether a semantically equivalent query is already cached.

        Steps:
          1. Predict fuzzy membership for the query embedding.
          2. Determine which cluster buckets to search.
          3. Compute cosine similarity against every entry in those buckets.
          4. If best similarity >= threshold: hit. Else: miss.
        """
        membership = predict_membership(
            query_embedding, self.centroids, self.fuzzifier
        )

        buckets_to_search = self._select_buckets(membership)

        best_similarity = -1.0
        best_entry: Optional[CacheEntry] = None

        for bucket_id in buckets_to_search:
            sim, entry = self._search_bucket(query_embedding, bucket_id)
            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        if best_similarity >= self.similarity_threshold and best_entry is not None:
            best_entry.hit_count += 1
            self._hit_count += 1
            logger.debug(
                f"Cache HIT: similarity={best_similarity:.4f}, "
                f"matched='{best_entry.query[:60]}...'"
            )
            return CacheLookupResult(
                hit=True,
                matched_query=best_entry.query,
                similarity_score=round(best_similarity, 4),
                result=best_entry.result,
                dominant_cluster=best_entry.dominant_cluster,
                entry=best_entry,
            )

        self._miss_count += 1
        logger.debug(
            f"Cache MISS: best_similarity={best_similarity:.4f} "
            f"(threshold={self.similarity_threshold})"
        )
        return CacheLookupResult(hit=False)

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: str,
    ) -> CacheEntry:
        """
        Store a new query-result pair in the cache.
        Assigns it to the dominant cluster bucket.
        """
        membership = predict_membership(
            query_embedding, self.centroids, self.fuzzifier
        )
        # Use _select_buckets (not argmax) so store and lookup always agree
        # on which bucket is "dominant". When all memberships are equal,
        # argmax returns 0 but argsort-based selection returns 14 — mismatch.
        dominant_cluster = self._select_buckets(membership)[0]

        entry = CacheEntry(
            query=query,
            embedding=query_embedding.copy(),
            result=result,
            dominant_cluster=dominant_cluster,
            membership_vector=membership,
        )

        self._buckets[dominant_cluster].append(entry)

        logger.debug(
            f"Stored in cluster {dominant_cluster}: '{query[:60]}' "
            f"(bucket size: {len(self._buckets[dominant_cluster])})"
        )
        return entry

    def flush(self) -> None:
        """Clear all cached entries and reset stats."""
        for k in self._buckets:
            self._buckets[k] = []
        self._hit_count = 0
        self._miss_count = 0
        logger.info("Cache flushed.")

    @property
    def stats(self) -> dict:
        """Return current cache statistics."""
        total = self._hit_count + self._miss_count
        total_entries = sum(len(b) for b in self._buckets.values())
        return {
            "total_entries": total_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self._hit_count / total, 4) if total > 0 else 0.0,
            "similarity_threshold": self.similarity_threshold,
            "bucket_sizes": {
                k: len(v) for k, v in self._buckets.items() if len(v) > 0
            },
        }

    def set_threshold(self, threshold: float) -> None:
        """
        Runtime threshold adjustment — lets you tune without restarting.
        Useful for A/B testing threshold values on live traffic.
        Note: changing threshold doesn't invalidate existing entries.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        old = self.similarity_threshold
        self.similarity_threshold = threshold
        logger.info(f"Cache threshold updated: {old:.2f} → {threshold:.2f}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_buckets(self, membership: np.ndarray) -> list[int]:
        """
        Choose which cluster buckets to search for a given membership vector.

        Strategy:
        - Always search the dominant cluster (highest membership).
        - If multi_cluster_lookup=True, also search the next (top_k_clusters-1)
          highest-membership clusters.

        Rationale: boundary documents (high entropy membership) are almost
        equally likely to have been stored in cluster A or cluster B when
        they were a miss. Multi-cluster lookup ensures we find them regardless
        of which bucket they landed in.

        Trade-off: searching k buckets means up to k × avg_bucket_size
        comparisons per lookup. With k=2, this is still O(2N/15) ≈ O(N/7.5),
        much better than O(N).
        """
        if self.multi_cluster_lookup:
            top_k = np.argsort(membership)[::-1][: self.top_k_clusters]
            return top_k.tolist()
        else:
            return [int(np.argmax(membership))]

    def _search_bucket(
        self,
        query_embedding: np.ndarray,
        bucket_id: int,
    ) -> tuple[float, Optional[CacheEntry]]:
        """
        Find the most similar cached entry in a single bucket.

        Cosine similarity between two L2-normalised vectors is just their
        dot product — O(dim) per comparison. Since MiniLM outputs normalised
        vectors (and we normalise at embed time), we use dot product directly.
        """
        bucket = self._buckets[bucket_id]
        if not bucket:
            return -1.0, None

        # Stack embeddings for vectorised dot products: (n_entries, dim)
        cached_embeddings = np.stack([e.embedding for e in bucket])

        # Dot product with query: (n_entries,)
        # Both sides are L2-normalised → this equals cosine similarity
        similarities = cached_embeddings @ query_embedding

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        return best_sim, bucket[best_idx]

    # ------------------------------------------------------------------
    # Threshold exploration (for analysis / testing)
    # ------------------------------------------------------------------

    def simulate_threshold_sweep(
        self,
        test_queries: list[tuple[str, np.ndarray]],  # [(query, embedding)]
        thresholds: list[float] = [0.70, 0.80, 0.85, 0.90, 0.95],
    ) -> list[dict]:
        """
        Given a list of test queries (not stored in the cache), report what
        the hit rate WOULD be at each threshold value.

        This is purely analytical — doesn't modify cache state.
        Useful for choosing θ before deployment.

        Returns: list of {"threshold": float, "hits": int, "hit_rate": float}
        """
        results = []

        for theta in thresholds:
            hits = 0
            for query, embedding in test_queries:
                membership = predict_membership(embedding, self.centroids, self.fuzzifier)
                buckets = self._select_buckets(membership)

                best_sim = -1.0
                for bid in buckets:
                    sim, _ = self._search_bucket(embedding, bid)
                    best_sim = max(best_sim, sim)

                if best_sim >= theta:
                    hits += 1

            hit_rate = hits / len(test_queries) if test_queries else 0.0
            results.append({
                "threshold": theta,
                "hits": hits,
                "total": len(test_queries),
                "hit_rate": round(hit_rate, 4),
            })
            logger.info(f"θ={theta}: {hits}/{len(test_queries)} hits ({hit_rate:.1%})")

        return results