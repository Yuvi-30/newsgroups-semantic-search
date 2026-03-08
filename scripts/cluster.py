"""
scripts/cluster.py
------------------
One-time script: load embeddings from ChromaDB → UMAP reduction →
run Fuzzy C-Means → update per-document cluster metadata → save artifacts.

Run from the project root AFTER scripts/ingest.py:
    python -m scripts.cluster

Expected runtime: ~10-20 minutes on CPU (UMAP is slower than PCA but
produces dramatically better cluster separation).

WHY UMAP OVER PCA:
  PCA is a linear reduction — it finds the directions of maximum variance
  but doesn't preserve the local neighbourhood structure that makes clusters
  separable. On 384-dim MiniLM embeddings, 50-component PCA retained only
  49% of variance and FCM produced uniform memberships (FPC = 1/k).

  UMAP is non-linear and explicitly optimises to keep nearby points together
  while pushing distant points apart — exactly what FCM needs. With 20 UMAP
  components, FPC typically rises to 0.3-0.7, indicating real cluster structure.

  Trade-off: UMAP is ~5x slower than PCA and is not trivially invertible,
  but for a one-time offline step that's acceptable.
"""

import logging
import os
import pickle
import sys

import numpy as np
from umap import UMAP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.clustering import (
    run_fuzzy_cmeans,
    save_clustering,
    get_cluster_category_distribution,
)
from src.vector_store import VectorStore

# UMAP hyperparameters
# n_components=20: enough dimensions for FCM to find structure,
#   few enough that curse of dimensionality doesn't return
# n_neighbors=15: controls local vs global structure trade-off;
#   15 is the UMAP default and works well for document embeddings
# min_dist=0.1: allows tighter clusters (0.0 = maximally tight,
#   1.0 = maximally spread); 0.1 is good for clustering tasks
UMAP_COMPONENTS = 20
UMAP_NEIGHBORS  = 15
UMAP_MIN_DIST   = 0.1
UMAP_MODEL_PATH = "./models/umap.pkl"

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    db_path      = os.getenv("CHROMA_DB_PATH", "./embeddings/chroma_db")
    collection   = os.getenv("CHROMA_COLLECTION", "newsgroups")
    model_name   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    n_clusters   = int(os.getenv("N_CLUSTERS", "15"))
    fuzzifier    = float(os.getenv("FUZZY_M", "2.0"))
    model_path   = os.getenv("CLUSTER_MODEL_PATH", "./models/fuzzy_clusters.pkl")

    # ── 1. Load embeddings ───────────────────────────────────────────
    store = VectorStore(db_path=db_path, collection_name=collection, model_name=model_name)
    if store.count() == 0:
        logger.error("ChromaDB is empty. Run scripts/ingest.py first.")
        sys.exit(1)

    doc_ids, embeddings = store.get_all_embeddings()
    logger.info(f"Loaded {len(doc_ids)} embeddings, shape: {embeddings.shape}")

    # ── 2. UMAP reduction ────────────────────────────────────────────
    logger.info(
        f"Running UMAP: {embeddings.shape[1]} → {UMAP_COMPONENTS} dims "
        f"(n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST})..."
    )
    logger.info("This takes ~10-20 minutes on CPU — please wait...")

    reducer = UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",      # matches how ChromaDB does similarity search
        random_state=42,
        verbose=True,
    )
    embeddings_reduced = reducer.fit_transform(embeddings).astype(np.float32)
    logger.info(f"UMAP complete. Reduced shape: {embeddings_reduced.shape}")

    # L2-normalise so dot product == cosine similarity in cache lookups
    norms = np.linalg.norm(embeddings_reduced, axis=1, keepdims=True)
    embeddings_reduced = embeddings_reduced / np.where(norms > 0, norms, 1)

    os.makedirs("./models", exist_ok=True)
    with open(UMAP_MODEL_PATH, "wb") as f:
        pickle.dump(reducer, f)
    logger.info(f"UMAP model saved to {UMAP_MODEL_PATH}")

    # ── 3. Fuzzy C-Means on UMAP-reduced embeddings ──────────────────
    result = run_fuzzy_cmeans(
        embeddings=embeddings_reduced,
        doc_ids=doc_ids,
        n_clusters=n_clusters,
        m=fuzzifier,
    )

    # ── 4. Validate cluster semantics ────────────────────────────────
    raw = store.collection.get(ids=doc_ids, include=["metadatas"])
    doc_categories = {
        doc_id: meta.get("category", "unknown")
        for doc_id, meta in zip(raw["ids"], raw["metadatas"])
    }
    distribution = get_cluster_category_distribution(result, doc_categories)

    logger.info("Cluster → top categories:")
    for cluster_id in range(result.n_clusters):
        cats = distribution[cluster_id]
        if not cats:
            continue
        top_cats = sorted(cats.items(), key=lambda x: -x[1])[:3]
        dominant_count = sum(1 for d in result.dominant_clusters if d == cluster_id)
        top_str = " | ".join(f"{c}: {p:.0%}" for c, p in top_cats)
        logger.info(f"  Cluster {cluster_id:2d} ({dominant_count:5d} docs): {top_str}")

    # ── 5. Save clustering artifact ───────────────────────────────────
    save_clustering(result, model_path)

    # ── 6. Update ChromaDB metadata ───────────────────────────────────
    logger.info("Writing cluster metadata back to ChromaDB...")
    updates = [
        {
            "doc_id": doc_ids[i],
            "dominant_cluster": int(result.dominant_clusters[i]),
            "memberships": result.membership_matrix[:, i].tolist(),
        }
        for i in range(len(doc_ids))
    ]
    store.bulk_update_cluster_metadata(updates)
    logger.info("All done.")


if __name__ == "__main__":
    main()