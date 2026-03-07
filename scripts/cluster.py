"""
scripts/cluster.py
------------------
One-time script: load embeddings from ChromaDB → PCA reduction →
run Fuzzy C-Means → update per-document cluster metadata → save artifacts.

Run from the project root AFTER scripts/ingest.py:
    python -m scripts.cluster

WHY PCA BEFORE FCM:
  FCM on raw 384-dim embeddings produces FPC = 1/k (the theoretical minimum,
  meaning maximally uniform memberships). This is the curse of dimensionality:
  in high-dimensional spaces all pairwise Euclidean distances converge to the
  same value, so FCM cannot distinguish near from far. Reducing to 50 dims
  via PCA retains >80% of variance while making distances meaningful again.
"""

import logging
import os
import pickle
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.clustering import (
    run_fuzzy_cmeans,
    save_clustering,
    get_cluster_category_distribution,
)
from src.vector_store import VectorStore

PCA_COMPONENTS = 50
PCA_MODEL_PATH = "./models/pca.pkl"

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    db_path = os.getenv("CHROMA_DB_PATH", "./embeddings/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "newsgroups")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    n_clusters = int(os.getenv("N_CLUSTERS", "15"))
    fuzzifier = float(os.getenv("FUZZY_M", "2.0"))
    model_path = os.getenv("CLUSTER_MODEL_PATH", "./models/fuzzy_clusters.pkl")

    store = VectorStore(db_path=db_path, collection_name=collection_name, model_name=model_name)
    if store.count() == 0:
        logger.error("ChromaDB is empty. Run scripts/ingest.py first.")
        sys.exit(1)

    # 1. Load embeddings
    doc_ids, embeddings = store.get_all_embeddings()
    logger.info(f"Loaded {len(doc_ids)} embeddings, shape: {embeddings.shape}")

    # 2. PCA reduction — fixes curse-of-dimensionality for FCM
    logger.info(f"Reducing {embeddings.shape[1]} → {PCA_COMPONENTS} dims via PCA...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    os.makedirs("./models", exist_ok=True)
    with open(PCA_MODEL_PATH, "wb") as f:
        pickle.dump(pca, f)
    logger.info(f"PCA model saved to {PCA_MODEL_PATH}")

    # 3. Fuzzy C-Means on reduced embeddings
    result = run_fuzzy_cmeans(
        embeddings=embeddings_reduced,
        doc_ids=doc_ids,
        n_clusters=n_clusters,
        m=fuzzifier,
    )

    # 4. Validate cluster semantics
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

    # 5. Save clustering artifact
    save_clustering(result, model_path)

    # 6. Update ChromaDB metadata
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
    logger.info("Done.")


if __name__ == "__main__":
    main()