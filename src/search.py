"""
search.py
---------
Orchestrates the full query pipeline with explicit PCA projection.

The projection happens HERE, not inside the cache, so the cache always
receives clean, normalised, same-dimensional vectors. This avoids the
inheritance / dimension-mismatch bugs that caused silent failures.
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.cache import SemanticCache, CacheLookupResult
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SearchResponse:
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: str
    dominant_cluster: int
    top_documents: list


def _format_result(vector_results: list[dict], query: str) -> str:
    if not vector_results:
        return "No relevant documents found."
    top = vector_results[0]
    category = top["metadata"].get("category", "unknown")
    cluster = top["metadata"].get("dominant_cluster", -1)
    text_snippet = top["text"][:500].replace("\n", " ").strip()
    lines = [
        f"Top match (similarity: {top['similarity']:.3f})",
        f"Category: {category} | Cluster: {cluster}",
        f"Excerpt: {text_snippet}",
        "",
        f"Found {len(vector_results)} relevant documents.",
    ]
    if len(vector_results) > 1:
        other_cats = set(r["metadata"].get("category", "?") for r in vector_results[1:])
        lines.append("Other relevant categories: " + ", ".join(other_cats))
    return "\n".join(lines)


def _project_and_normalize(embedding: np.ndarray, pca) -> np.ndarray:
    """
    Project a 384-dim embedding to PCA space and L2-normalise.
    Returns a unit vector in the reduced space.
    Both lookup and store use this — guarantees dot product == cosine similarity.
    """
    reduced = pca.transform(embedding.reshape(1, -1))[0].astype(np.float32)
    norm = np.linalg.norm(reduced)
    return reduced / norm if norm > 0 else reduced


class SearchService:
    """
    Stateful search service. Instantiated once at FastAPI startup.
    Holds the PCA model explicitly so projection is transparent and debuggable.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        cache: SemanticCache,
        pca,                    # sklearn PCA fitted on corpus embeddings
        n_results: int = 10,
    ):
        self.vector_store = vector_store
        self.cache = cache
        self.pca = pca
        self.n_results = n_results

    def query(self, query_text: str) -> SearchResponse:
        logger.info(f"Query: '{query_text[:80]}'")

        # 1. Embed in full 384-dim space (used for vector DB search)
        raw_embedding = self.vector_store.embed_query(query_text)

        # 2. Project + normalise to 50-dim for cache operations
        cache_embedding = _project_and_normalize(raw_embedding, self.pca)

        logger.debug(f"raw_embedding norm: {np.linalg.norm(raw_embedding):.4f}, "
                     f"cache_embedding norm: {np.linalg.norm(cache_embedding):.4f}")

        # 3. Cache lookup (uses 50-dim normalised vector)
        cache_result: CacheLookupResult = self.cache.lookup(query_text, cache_embedding)

        if cache_result.hit:
            logger.info(
                f"Cache HIT — matched: '{cache_result.matched_query[:60]}' "
                f"(sim={cache_result.similarity_score:.4f})"
            )
            return SearchResponse(
                query=query_text,
                cache_hit=True,
                matched_query=cache_result.matched_query,
                similarity_score=cache_result.similarity_score,
                result=cache_result.result,
                dominant_cluster=cache_result.dominant_cluster,
                top_documents=[],
            )

        # 4. Vector search with full 384-dim embedding
        logger.info("Cache MISS — running vector search")
        vector_results = self.vector_store.query(
            query_embedding=raw_embedding,
            n_results=self.n_results,
        )

        result_text = _format_result(vector_results, query_text)

        dominant_cluster = -1
        if vector_results:
            dominant_cluster = vector_results[0]["metadata"].get("dominant_cluster", -1)

        # 5. Store in cache using 50-dim normalised vector
        self.cache.store(
            query=query_text,
            query_embedding=cache_embedding,
            result=result_text,
        )

        logger.info(
            f"Stored in cache. Stats: {self.cache.stats['total_entries']} entries, "
            f"{self.cache.stats['miss_count']} misses"
        )

        return SearchResponse(
            query=query_text,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=result_text,
            dominant_cluster=dominant_cluster,
            top_documents=vector_results[:5],
        )

    @property
    def cache_stats(self) -> dict:
        return self.cache.stats

    def flush_cache(self) -> None:
        self.cache.flush()

    def set_cache_threshold(self, threshold: float) -> None:
        self.cache.set_threshold(threshold)