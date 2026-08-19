import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from google import genai

from constructor_agent_tools.settings import settings
from constructor_agent_tools.bundle.dynamic_schemas import CatalogProduct

logger = logging.getLogger("constructor_agent_tools.bundle.embedding_engine")

DEFAULT_STORE_DIR = Path("data/vector_store")


class VectorStore:
    """
    In-memory vector store backed by NumPy arrays with disk persistence.
    Supports adding vectors, searching by cosine similarity, and saving/loading to disk.
    """

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or DEFAULT_STORE_DIR
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None  # Shape: (N, D)
        self._metadata: Dict[str, dict] = {}  # product_id -> serialized product data
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load from disk on first access."""
        if not self._loaded:
            self.load()
            self._loaded = True

    def add(self, product_id: str, vector: List[float], metadata: Optional[dict] = None):
        """Add a product vector to the store."""
        self._ensure_loaded()
        vec = np.array(vector, dtype=np.float32)

        if self._vectors is None:
            self._vectors = vec.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])

        self._ids.append(product_id)
        if metadata:
            self._metadata[product_id] = metadata

    def add_batch(self, product_ids: List[str], vectors: List[List[float]], metadata_list: Optional[List[dict]] = None):
        """Add multiple product vectors at once."""
        self._ensure_loaded()
        new_vecs = np.array(vectors, dtype=np.float32)

        if self._vectors is None:
            self._vectors = new_vecs
        else:
            self._vectors = np.vstack([self._vectors, new_vecs])

        self._ids.extend(product_ids)
        if metadata_list:
            for pid, meta in zip(product_ids, metadata_list):
                self._metadata[pid] = meta

    def search(self, query_vector: List[float], top_k: int = 50, category_filter: Optional[str] = None) -> List[Tuple[str, float, dict]]:
        """
        Search for the top_k most similar products by cosine similarity.
        Returns list of (product_id, similarity_score, metadata).
        """
        self._ensure_loaded()

        if self._vectors is None or len(self._ids) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)

        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        norms = np.linalg.norm(self._vectors, axis=1)
        query_norm = np.linalg.norm(query)

        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1e-10, norms)
        safe_query_norm = max(query_norm, 1e-10)

        similarities = np.dot(self._vectors, query) / (safe_norms * safe_query_norm)

        # Apply category filter if provided
        if category_filter:
            mask = np.array([
                self._metadata.get(pid, {}).get("category", "") == category_filter
                for pid in self._ids
            ])
            similarities = np.where(mask, similarities, -1.0)

        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0:
                break  # No more positive matches
            pid = self._ids[idx]
            meta = self._metadata.get(pid, {})
            results.append((pid, score, meta))

        return results

    def save(self):
        """Persist vectors, IDs, and metadata to disk."""
        self.store_dir.mkdir(parents=True, exist_ok=True)

        if self._vectors is not None:
            np.save(str(self.store_dir / "vectors.npy"), self._vectors)

        with open(self.store_dir / "ids.json", "w") as f:
            json.dump(self._ids, f)

        with open(self.store_dir / "metadata.json", "w") as f:
            json.dump(self._metadata, f)

        logger.info(f"VectorStore saved to {self.store_dir} ({len(self._ids)} products)")

    def load(self):
        """Load vectors, IDs, and metadata from disk."""
        vectors_path = self.store_dir / "vectors.npy"
        ids_path = self.store_dir / "ids.json"
        metadata_path = self.store_dir / "metadata.json"

        if vectors_path.exists() and ids_path.exists():
            self._vectors = np.load(str(vectors_path))
            with open(ids_path, "r") as f:
                self._ids = json.load(f)
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self._metadata = json.load(f)
            logger.info(f"VectorStore loaded from {self.store_dir} ({len(self._ids)} products)")
        else:
            self._ids = []
            self._vectors = None
            self._metadata = {}
            logger.info("No existing VectorStore found, starting empty.")

    @property
    def size(self) -> int:
        self._ensure_loaded()
        return len(self._ids)


class EmbeddingEngine:
    """
    Handles embedding generation using Gemini's text-embedding model.
    Call 2: Batch-embeds rule descriptions and catalog products.
    """

    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name
        self._client = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings using Gemini's embedding model.
        This is Call 2 in the architecture.
        """
        result = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    async def embed_catalog(self, products: List[CatalogProduct], vector_store: VectorStore):
        """
        Pre-compute embeddings for a batch of catalog products and add them to the vector store.
        Embeds the concatenation of title + description for each product.
        """
        texts = [f"{p.title}. {p.description}" for p in products]
        embeddings = await self.embed_texts(texts)

        product_ids = [p.id for p in products]
        metadata_list = [
            {
                "id": p.id,
                "title": p.title,
                "description": p.description,
                "category": p.category,
                "price": p.price,
                "rating": p.rating,
                "image_url": p.image_url,
            }
            for p in products
        ]

        vector_store.add_batch(product_ids, embeddings, metadata_list)
        vector_store.save()
        logger.info(f"Embedded and stored {len(products)} products.")
