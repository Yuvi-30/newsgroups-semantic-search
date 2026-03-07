"""
scripts/ingest.py
-----------------
One-time script: download 20 Newsgroups → clean → embed → persist in ChromaDB.

Run from the project root:
    python -m scripts.ingest

Expected runtime: ~5-10 minutes on CPU (embedding 18k documents with MiniLM).
Output: populated ChromaDB at ./embeddings/chroma_db/
"""

import logging
import os
import sys

# Allow running as `python -m scripts.ingest` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from sklearn.datasets import fetch_20newsgroups

from src.preprocessing import preprocess_corpus
from src.vector_store import VectorStore

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ----------------------------------------------------------------
    # 1. Load dataset via sklearn (downloads and caches automatically)
    # ----------------------------------------------------------------
    logger.info("Fetching 20 Newsgroups dataset (train + test splits)...")

    # We load BOTH train and test splits to maximise corpus size.
    # remove=() means we keep everything including headers for our own
    # cleaning pipeline — sklearn's built-in removal is too aggressive
    # (strips content as well as metadata in edge cases).
    newsgroups = fetch_20newsgroups(
        subset="all",          # train + test
        remove=(),             # we handle our own cleaning
        shuffle=False,
    )

    logger.info(
        f"Loaded {len(newsgroups.data)} documents "
        f"across {len(newsgroups.target_names)} categories"
    )

    # ----------------------------------------------------------------
    # 2. Build raw document list
    # ----------------------------------------------------------------
    raw_docs = []
    for i, (text, target_idx) in enumerate(zip(newsgroups.data, newsgroups.target)):
        raw_docs.append({
            "id": f"doc_{i:05d}",
            "text": text,
            "category": newsgroups.target_names[target_idx],
        })

    # ----------------------------------------------------------------
    # 3. Preprocess
    # ----------------------------------------------------------------
    logger.info("Preprocessing corpus...")
    cleaned_docs, stats = preprocess_corpus(raw_docs)

    logger.info("Preprocessing stats:")
    for k, v in stats.items():
        if k != "category_distribution":
            logger.info(f"  {k}: {v}")

    logger.info("Category distribution after cleaning:")
    for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
        logger.info(f"  {cat:45s}: {count:5d}")

    # ----------------------------------------------------------------
    # 4. Ingest into ChromaDB
    # ----------------------------------------------------------------
    db_path = os.getenv("CHROMA_DB_PATH", "./embeddings/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "newsgroups")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    store = VectorStore(
        db_path=db_path,
        collection_name=collection_name,
        model_name=model_name,
    )

    # Check if already ingested — skip to avoid re-embedding everything
    if store.count() >= len(cleaned_docs) * 0.95:
        logger.info(
            f"Collection already contains {store.count()} documents. "
            "Skipping ingestion. Delete ./embeddings/chroma_db to re-ingest."
        )
        return

    n_ingested = store.ingest(cleaned_docs)
    logger.info(f"Ingestion complete: {n_ingested} documents stored.")
    logger.info(f"ChromaDB collection size: {store.count()}")


if __name__ == "__main__":
    main()
