"""
api/main.py — FastAPI service.
PCA projection is handled in SearchService, not the cache.
Cache always receives clean, normalised, 50-dim vectors.
"""

import logging
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.cache import SemanticCache
from src.clustering import load_clustering
from src.search import SearchService
from src.vector_store import VectorStore

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: str
    dominant_cluster: int

class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float

class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    db_path      = os.getenv("CHROMA_DB_PATH", "./embeddings/chroma_db")
    collection   = os.getenv("CHROMA_COLLECTION", "newsgroups")
    model_name   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    cluster_path = os.getenv("CLUSTER_MODEL_PATH", "./models/fuzzy_clusters.pkl")
    pca_path     = "./models/pca.pkl"
    threshold    = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))
    fuzzifier    = float(os.getenv("FUZZY_M", "2.0"))

    for path, label in [
        (db_path,      "ChromaDB"),
        (cluster_path, "Clustering model"),
        (pca_path,     "PCA model"),
    ]:
        if not os.path.exists(path):
            raise RuntimeError(f"{label} not found at '{path}'. Run setup scripts first.")

    # Load all components
    vector_store = VectorStore(db_path=db_path, collection_name=collection, model_name=model_name)
    logger.info(f"Vector store: {vector_store.count()} documents")

    clustering = load_clustering(cluster_path)
    logger.info(f"Clustering: {clustering.n_clusters} clusters, FPC={clustering.partition_coefficient:.4f}, centroid shape={clustering.centroids.shape}")

    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    logger.info(f"PCA: {pca.n_components_} components, {pca.explained_variance_ratio_.sum():.1%} variance explained")

    # Sanity check: centroid dims must match PCA output dims
    assert clustering.centroids.shape[1] == pca.n_components_, (
        f"Centroid dim {clustering.centroids.shape[1]} != PCA components {pca.n_components_}. "
        "Re-run scripts/cluster.py."
    )

    cache = SemanticCache(
        centroids=clustering.centroids,
        fuzzifier=fuzzifier,
        similarity_threshold=threshold,
        multi_cluster_lookup=True,
        top_k_clusters=2,
    )

    # SearchService owns the PCA model and handles projection explicitly
    search_service = SearchService(
        vector_store=vector_store,
        cache=cache,
        pca=pca,
        n_results=10,
    )

    app.state.search_service = search_service
    app.state.clustering      = clustering

    logger.info(f"Service ready. Threshold={threshold}")
    yield
    logger.info("Shutting down.")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Newsgroups Semantic Search",
    description="Semantic search with fuzzy cluster-aware caching over 20 Newsgroups.",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, body: QueryRequest):
    svc: SearchService = request.app.state.search_service
    try:
        response = svc.query(body.query)
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query=response.query,
        cache_hit=response.cache_hit,
        matched_query=response.matched_query,
        similarity_score=response.similarity_score,
        result=response.result,
        dominant_cluster=response.dominant_cluster,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(request: Request):
    stats = request.app.state.search_service.cache_stats
    return CacheStatsResponse(
        total_entries=stats["total_entries"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"],
    )


@app.delete("/cache")
async def flush_cache(request: Request):
    request.app.state.search_service.flush_cache()
    return {"message": "Cache flushed.", "total_entries": 0}


@app.post("/cache/threshold")
async def set_threshold(request: Request, body: ThresholdRequest):
    try:
        request.app.state.search_service.set_cache_threshold(body.threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": f"Threshold updated to {body.threshold}"}


@app.get("/cluster/info")
async def cluster_info(request: Request):
    clustering = request.app.state.clustering
    svc: SearchService = request.app.state.search_service
    return {
        "n_clusters":            clustering.n_clusters,
        "fuzzifier":             clustering.fuzzifier,
        "partition_coefficient": round(clustering.partition_coefficient, 4),
        "centroid_shape":        list(clustering.centroids.shape),
        "n_documents":           len(clustering.doc_ids),
        "cache_bucket_sizes":    svc.cache_stats.get("bucket_sizes", {}),
    }