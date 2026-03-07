"""
clustering.py
-------------
Fuzzy C-Means clustering over the document embedding space.

WHY FUZZY C-MEANS over hard alternatives:
  - KMeans would give every document exactly one cluster label. A post about
    "gun control legislation" sits equally between rec.guns and talk.politics.misc —
    hard assignment loses this signal entirely.
  - Fuzzy C-Means (FCM) assigns each document a *membership vector* summing to 1.
    [0.0, 0.05, 0.72, 0.0, 0.23, ...] — the document mostly belongs to cluster 2,
    meaningfully overlaps cluster 4, and is essentially absent from the rest.
  - This matters downstream: the cache lookup uses membership vectors to partition
    the search space, and documents at cluster boundaries naturally appear in
    multiple partitions.

WHY 15 CLUSTERS (not 20):
  - The 20 newsgroup categories are not semantically orthogonal:
      comp.sys.ibm.pc.hardware / comp.sys.mac.hardware / comp.os.ms-windows.misc
      all cluster together; similarly talk.religion.misc / soc.religion.christian.
  - Elbow analysis on inertia + silhouette scores peaks at ~12-16; we pick 15
    as a round number inside that range that preserves meaningful boundary cases.
  - See notebooks/cluster_analysis.ipynb for the full sweep.

FUZZIFIER m=2.0:
  - The standard FCM default. m=1.0 collapses to hard K-Means; m→∞ assigns
    uniform membership to everything. m=2.0 gives sharp-but-not-binary assignments.
  - Higher m (e.g. 2.5) is worth exploring if clusters look too clean.

NORMALISATION:
  - MiniLM embeddings are already L2-normalised. FCM with cosine distance would
    be ideal, but skfuzzy implements Euclidean FCM. On a unit hypersphere,
    Euclidean distance is monotone with cosine distance, so results are equivalent.
"""

import logging
import os
import pickle
from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from tqdm import tqdm

logger = logging.getLogger(__name__)

# FCM convergence criteria
MAX_ITER = 300
FCM_ERROR = 1e-6   # stop when centroid shift drops below this


@dataclass
class ClusteringResult:
    """All outputs from a fuzzy clustering run."""
    n_clusters: int
    fuzzifier: float
    # (n_clusters, n_docs) — each column is one document's membership vector
    membership_matrix: np.ndarray
    # (n_clusters, embedding_dim) — cluster centroids
    centroids: np.ndarray
    # Per-document dominant cluster (argmax of membership vector)
    dominant_clusters: np.ndarray
    # Partition coefficient — 1.0 = hard, 1/n_clusters = maximally fuzzy
    partition_coefficient: float
    doc_ids: list[str]


def run_fuzzy_cmeans(
    embeddings: np.ndarray,    # (n_docs, embedding_dim)
    doc_ids: list[str],
    n_clusters: int = 15,
    m: float = 2.0,
    seed: int = 42,
) -> ClusteringResult:
    """
    Run Fuzzy C-Means on the embedding matrix.

    skfuzzy.cmeans expects data shape (n_features, n_samples) — transposed
    from the conventional (n_samples, n_features). We transpose in/out.
    """
    logger.info(
        f"Running Fuzzy C-Means: {n_clusters} clusters, m={m}, "
        f"{embeddings.shape[0]} documents, {embeddings.shape[1]} dims"
    )

    # skfuzzy shape convention: (n_features, n_samples)
    data = embeddings.T.astype(np.float64)

    np.random.seed(seed)

    # cmeans returns:
    # cntr       — centroids (n_clusters, n_features)
    # u          — membership matrix (n_clusters, n_samples)
    # u0         — initial membership
    # d          — distance matrix
    # jm         — objective function history
    # p          — number of iterations
    # fpc        — fuzzy partition coefficient
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=data,
        c=n_clusters,
        m=m,
        error=FCM_ERROR,
        maxiter=MAX_ITER,
        seed=seed,
    )

    dominant = np.argmax(u, axis=0)   # shape: (n_docs,)

    logger.info(
        f"FCM converged in {p} iterations. "
        f"Fuzzy Partition Coefficient: {fpc:.4f} "
        f"(1.0=hard, {1/n_clusters:.4f}=max fuzzy)"
    )

    # Log cluster size distribution by dominant assignment
    unique, counts = np.unique(dominant, return_counts=True)
    for c_id, count in zip(unique, counts):
        logger.info(f"  Cluster {c_id:2d}: {count:5d} dominant documents")

    return ClusteringResult(
        n_clusters=n_clusters,
        fuzzifier=m,
        membership_matrix=u,          # (n_clusters, n_docs)
        centroids=cntr,               # (n_clusters, n_dims)
        dominant_clusters=dominant,   # (n_docs,)
        partition_coefficient=fpc,
        doc_ids=doc_ids,
    )


def sweep_cluster_counts(
    embeddings: np.ndarray,
    doc_ids: list[str],
    k_range: range = range(8, 22),
    m: float = 2.0,
    seed: int = 42,
) -> list[dict]:
    """
    Sweep over different k values and record metrics to justify k selection.
    Returns list of {"k": int, "fpc": float, "inertia": float} for plotting.
    Used in notebooks/cluster_analysis.ipynb.
    """
    results = []
    data = embeddings.T.astype(np.float64)

    for k in tqdm(k_range, desc="Sweeping cluster counts"):
        np.random.seed(seed)
        _, u, _, d, jm, p, fpc = fuzz.cluster.cmeans(
            data=data, c=k, m=m, error=FCM_ERROR, maxiter=MAX_ITER, seed=seed,
        )
        # Inertia: sum of (membership^m * distance^2) — the FCM objective
        inertia = float(jm[-1])
        results.append({"k": k, "fpc": fpc, "inertia": inertia, "iterations": p})
        logger.info(f"k={k}: fpc={fpc:.4f}, inertia={inertia:.2f}, iters={p}")

    return results


def save_clustering(result: ClusteringResult, path: str) -> None:
    """Pickle the clustering result for reuse across process restarts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Clustering saved to {path}")


def load_clustering(path: str) -> ClusteringResult:
    """Load a previously saved clustering result."""
    with open(path, "rb") as f:
        result = pickle.load(f)
    logger.info(
        f"Loaded clustering: {result.n_clusters} clusters, "
        f"{len(result.doc_ids)} documents"
    )
    return result


# ------------------------------------------------------------------
# Analysis utilities (used in notebooks and cluster.py script)
# ------------------------------------------------------------------

def get_cluster_top_docs(
    result: ClusteringResult,
    cluster_id: int,
    top_n: int = 10,
) -> list[dict]:
    """
    Return the top_n documents most strongly belonging to cluster_id.
    High membership = document is a clear exemplar of the cluster.
    """
    memberships = result.membership_matrix[cluster_id]  # (n_docs,)
    top_indices = np.argsort(memberships)[::-1][:top_n]
    return [
        {
            "doc_id": result.doc_ids[i],
            "membership": float(memberships[i]),
        }
        for i in top_indices
    ]


def get_boundary_docs(
    result: ClusteringResult,
    top_n: int = 20,
    entropy_weighted: bool = True,
) -> list[dict]:
    """
    Return documents that most straddle cluster boundaries.

    Two operationalisations:
    1. entropy_weighted=True: sort by Shannon entropy of membership vector.
       High entropy = uncertainty spread across many clusters.
    2. entropy_weighted=False: sort by (1 - max_membership).
       Maximises documents where even the dominant cluster is weak.

    Both capture different aspects of boundary-ness. The entropy version
    is more sensitive to multi-cluster overlap (a document in 3 clusters
    equally outranks one in 2 clusters equally). The max version picks up
    documents that are genuinely between exactly two clusters.
    """
    u = result.membership_matrix.T    # (n_docs, n_clusters)
    # Clip to avoid log(0)
    u_safe = np.clip(u, 1e-10, 1.0)

    if entropy_weighted:
        # Shannon entropy — high entropy = maximally uncertain
        entropy = -np.sum(u_safe * np.log(u_safe), axis=1)
        scores = entropy
    else:
        max_membership = np.max(u, axis=0)
        scores = 1.0 - max_membership

    top_indices = np.argsort(scores)[::-1][:top_n]

    return [
        {
            "doc_id": result.doc_ids[i],
            "score": float(scores[i]),
            "dominant_cluster": int(result.dominant_clusters[i]),
            "membership_vector": [round(float(v), 4) for v in u[i]],
        }
        for i in top_indices
    ]


def get_cluster_category_distribution(
    result: ClusteringResult,
    doc_categories: dict[str, str],   # {doc_id: category}
) -> dict[int, dict[str, float]]:
    """
    For each cluster, compute what fraction of its dominant documents
    come from each 20NG category. High purity = cluster is semantically coherent.
    This is our primary sanity check: if a cluster is a random mix of categories,
    the embeddings (or clustering) have failed.
    """
    distributions: dict[int, dict[str, int]] = {k: {} for k in range(result.n_clusters)}

    for i, doc_id in enumerate(result.doc_ids):
        cluster = int(result.dominant_clusters[i])
        category = doc_categories.get(doc_id, "unknown")
        distributions[cluster][category] = distributions[cluster].get(category, 0) + 1

    # Normalise to fractions
    return {
        k: {cat: count / sum(cats.values()) for cat, count in cats.items()}
        for k, cats in distributions.items()
    }


def predict_membership(
    query_embedding: np.ndarray,    # (embedding_dim,)
    centroids: np.ndarray,          # (n_clusters, embedding_dim)
    m: float = 2.0,
) -> np.ndarray:
    """
    Compute fuzzy membership vector for a new (unseen) query embedding.

    FCM membership for point x given centroids {c_k}:
        u_k = 1 / sum_j [ (||x - c_k|| / ||x - c_j||)^(2/(m-1)) ]

    This is the standard FCM prediction formula — equivalent to what
    cmeans_predict does but without the skfuzzy overhead for a single vector.
    """
    # Euclidean distances to each centroid
    diffs = centroids - query_embedding[np.newaxis, :]  # (n_clusters, dim)
    distances = np.linalg.norm(diffs, axis=1)           # (n_clusters,)

    # Handle the degenerate case where query exactly hits a centroid
    if np.any(distances == 0):
        membership = np.zeros(len(centroids))
        membership[distances == 0] = 1.0
        return membership

    # FCM formula: u_k proportional to (1 / d_k)^(2/(m-1))
    exponent = 2.0 / (m - 1.0)
    inv_dist = (1.0 / distances) ** exponent
    membership = inv_dist / inv_dist.sum()

    return membership.astype(np.float32)
