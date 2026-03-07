"""
vector_store.py
---------------
ChromaDB wrapper for ingesting, storing, and querying document embeddings.

Design decisions:

1. WHY CHROMADB
   - Persistent local storage — no server process needed for development or CI.
   - Native metadata filtering lets us pre-filter by cluster or category before
     doing cosine search, which is exactly what the cache lookup needs.
   - Straightforward Python API without the operational overhead of Weaviate/Qdrant.

2. EMBEDDING MODEL: all-MiniLM-L6-v2
   - 384 dimensions — small enough that 20k documents fit comfortably in RAM.
   - Trained on 1B+ sentence pairs; strong on short-to-medium English text.
   - Symmetric model: query and document embeddings are in the same space
     (critical for cosine similarity to be meaningful at retrieval time).
   - The alternative (all-mpnet-base-v2, 768-dim) would give ~2% better
     benchmark scores but 3x inference time and 4x memory — not worth it here.

3. BATCH INGESTION
   - We embed in batches of EMBED_BATCH_SIZE to avoid OOM on the GPU/CPU.
   - ChromaDB upsert is used so re-running ingestion is idempotent.

4. METADATA STORED PER DOCUMENT
   - category: original newsgroup label (for evaluation)
   - dominant_cluster: the argmax of the fuzzy membership vector (added later)
   - cluster_memberships: JSON-serialised soft distribution (added later)
   This means a single ChromaDB query returns everything needed for a response.
"""

import json
import logging
import os
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.preprocessing import CleanedDocument

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 256   # tune down if RAM-constrained; 256 is safe for MiniLM


class VectorStore:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name

        # Load embedding model once; kept alive for the process lifetime.
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Persistent ChromaDB client — data survives process restarts.
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # get_or_create so both first run and subsequent runs work cleanly.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # cosine distance is standard for sentence embeddings;
            # dot product would require normalised vectors (which MiniLM outputs,
            # but cosine is explicit and safer if we ever swap models).
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Collection '{collection_name}' ready. "
            f"Current size: {self.collection.count()} documents"
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts. Returns float32 array of shape (N, dim).
        normalize_embeddings=True ensures cosine similarity == dot product,
        which is what ChromaDB's cosine space uses internally.
        """
        return self.model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns 1-D float32 array."""
        return self.embed([query])[0]

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[CleanedDocument], batch_size: int = 512) -> int:
        """
        Embed and upsert documents into ChromaDB.
        Uses upsert (not add) so re-runs are idempotent.

        Returns the number of documents successfully ingested.
        """
        if not documents:
            logger.warning("No documents to ingest.")
            return 0

        logger.info(f"Ingesting {len(documents)} documents in batches of {batch_size}...")
        ingested = 0

        for start in tqdm(range(0, len(documents), batch_size), desc="Ingesting"):
            batch = documents[start : start + batch_size]

            texts = [doc.text for doc in batch]
            ids = [doc.doc_id for doc in batch]
            metadatas = [
                {
                    "category": doc.category,
                    "cleaned_length": doc.cleaned_length,
                    # Placeholder — filled in by clustering step.
                    # Storing here means we don't need a separate lookup table.
                    "dominant_cluster": -1,
                    "cluster_memberships": "[]",
                    # Store full (non-truncated) text for result display
                    "full_text": doc.full_text[:2000],  # ChromaDB metadata cap
                }
                for doc in batch
            ]

            embeddings = self.embed(texts).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            ingested += len(batch)

        logger.info(f"Ingestion complete. Total in collection: {self.collection.count()}")
        return ingested

    def update_cluster_metadata(
        self,
        doc_id: str,
        dominant_cluster: int,
        memberships: list[float],
    ) -> None:
        """
        Update a document's cluster metadata after fuzzy clustering.
        ChromaDB supports partial metadata updates via upsert.
        """
        # Fetch existing metadata to merge (ChromaDB requires full metadata on upsert)
        result = self.collection.get(ids=[doc_id], include=["metadatas", "embeddings", "documents"])
        if not result["ids"]:
            logger.warning(f"Document {doc_id} not found in collection.")
            return

        meta = result["metadatas"][0]
        meta["dominant_cluster"] = dominant_cluster
        meta["cluster_memberships"] = json.dumps(
            [round(float(m), 4) for m in memberships]
        )

        self.collection.upsert(
            ids=[doc_id],
            embeddings=result["embeddings"][0],
            documents=result["documents"][0],
            metadatas=[meta],
        )

    def bulk_update_cluster_metadata(
        self,
        updates: list[dict],  # [{"doc_id": str, "dominant_cluster": int, "memberships": list[float]}]
        batch_size: int = 512,
    ) -> None:
        """
        Efficient bulk update of cluster metadata.
        Fetches, merges, and upserts in batches to avoid N+1 round-trips.
        """
        logger.info(f"Updating cluster metadata for {len(updates)} documents...")

        for start in tqdm(range(0, len(updates), batch_size), desc="Updating clusters"):
            batch = updates[start : start + batch_size]
            ids = [u["doc_id"] for u in batch]

            result = self.collection.get(
                ids=ids,
                include=["metadatas", "embeddings", "documents"],
            )

            new_metadatas = []
            for i, meta in enumerate(result["metadatas"]):
                update = batch[i]
                meta["dominant_cluster"] = update["dominant_cluster"]
                meta["cluster_memberships"] = json.dumps(
                    [round(float(m), 4) for m in update["memberships"]]
                )
                new_metadatas.append(meta)

            self.collection.upsert(
                ids=result["ids"],
                embeddings=result["embeddings"],
                documents=result["documents"],
                metadatas=new_metadatas,
            )

        logger.info("Cluster metadata update complete.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search against the collection.

        Args:
            query_embedding: 1-D normalised embedding vector.
            n_results: number of nearest neighbours to return.
            where: optional ChromaDB metadata filter (e.g. {"dominant_cluster": 3})

        Returns:
            List of result dicts with id, text, metadata, distance, similarity.
        """
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(n_results, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)

        results = []
        for i, doc_id in enumerate(raw["ids"][0]):
            # ChromaDB cosine space returns cosine *distance* (1 - similarity)
            distance = raw["distances"][0][i]
            similarity = 1.0 - distance

            results.append({
                "id": doc_id,
                "text": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": round(distance, 4),
                "similarity": round(similarity, 4),
            })

        return results

    def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """
        Retrieve all document IDs and their embeddings.
        Used by the clustering step to build the membership matrix.
        ChromaDB loads everything into memory — fine for 20k × 384.
        """
        logger.info("Fetching all embeddings from ChromaDB...")
        result = self.collection.get(include=["embeddings"])
        ids = result["ids"]
        embeddings = np.array(result["embeddings"], dtype=np.float32)
        logger.info(f"Fetched {len(ids)} embeddings, shape: {embeddings.shape}")
        return ids, embeddings

    def count(self) -> int:
        return self.collection.count()
