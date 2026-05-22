"""
Semantic Search Engine — Core

Uses sentence-transformers for embeddings + FAISS for approximate nearest-neighbour search.
"""
import numpy as np
import faiss
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class SearchResult:
    document: Document
    score: float   # cosine similarity, 0–1 (higher = more similar)
    rank: int


class EmbeddingModel:
    """
    Wraps sentence-transformers for text → vector conversion.

    Model choice: all-MiniLM-L6-v2
    - 384 dimensions (vs 768 for large models)
    - ~5x faster inference, 90% of the quality
    - Fits in 90MB RAM — can run on CPU in production for low traffic

    For GPU inference at scale: batch size=256, use sentence_transformers with CUDA
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.dim = 384

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Returns float32 array of shape (len(texts), dim).
        Normalized to unit length — enables cosine similarity via dot product.
        """
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # L2 normalize → dot product = cosine similarity
            show_progress_bar=len(texts) > 100
        )
        return embeddings.astype(np.float32)


class FAISSIndex:
    """
    FAISS vector index for approximate nearest-neighbour (ANN) search.

    Index type progression:
    - <10K docs:    IndexFlatIP (exact, brute force) — simple, no training needed
    - 10K–1M docs:  IndexIVFFlat (inverted file, ~10x faster) — needs training
    - >1M docs:     IndexIVFPQ (product quantization, 10x memory reduction) — lossy but scalable

    We use IndexFlatIP here for correctness. Switch to IVFFlat for scale.
    IP = inner product. Since vectors are L2-normalized, IP == cosine similarity.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # exact search, cosine via IP
        self._id_map: List[str] = []           # FAISS int index → document ID

    def add(self, embeddings: np.ndarray, doc_ids: List[str]):
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings)
        self._id_map.extend(doc_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns list of (doc_id, cosine_score) sorted by score descending.
        Scores in range [-1, 1]; for well-separated topics usually 0.3–0.95.
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def size(self) -> int:
        return self.index.ntotal

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.ids.json", "w") as f:
            json.dump(self._id_map, f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.ids.json") as f:
            self._id_map = json.load(f)


class SemanticSearchEngine:
    """
    Full pipeline: text → embed → index → search → results.
    """
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.index = FAISSIndex(dim=384)
        self._documents: Dict[str, Document] = {}

    def add_documents(self, documents: List[Document]) -> int:
        texts = [doc.text for doc in documents]
        embeddings = self.embedder.encode(texts)
        doc_ids = [doc.id for doc in documents]

        self.index.add(embeddings, doc_ids)
        for doc in documents:
            self._documents[doc.id] = doc

        return len(documents)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if self.index.size() == 0:
            return []

        query_embedding = self.embedder.encode([query])
        raw_results = self.index.search(query_embedding, top_k=top_k)

        results = []
        for rank, (doc_id, score) in enumerate(raw_results, start=1):
            doc = self._documents.get(doc_id)
            if doc:
                results.append(SearchResult(document=doc, score=score, rank=rank))
        return results

    # TODO: Implement hybrid search
    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.7):
        """
        Combine semantic + keyword search using Reciprocal Rank Fusion.
        alpha: weight for semantic score (1-alpha for BM25 keyword score)

        Steps:
        1. Run semantic search → get ranked list A
        2. Run BM25 keyword search (use rank_bm25 library) → get ranked list B
        3. RRF score = alpha * (1/(k+rank_A)) + (1-alpha) * (1/(k+rank_B)), k=60
        4. Re-sort by RRF score
        """
        raise NotImplementedError("Implement hybrid BM25 + semantic search")
