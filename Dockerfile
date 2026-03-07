FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding model so the container doesn't need internet at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

# The data pipeline (ingest + cluster) must be run before building this image,
# so the embeddings/ and models/ directories are present in the build context.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
