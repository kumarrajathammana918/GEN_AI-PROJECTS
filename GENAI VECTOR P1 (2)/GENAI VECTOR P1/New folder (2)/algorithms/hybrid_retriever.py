from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from .base_retriever import BaseRetriever


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score < 1e-9:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining BM25 (lexical) and cosine similarity (semantic-ish TF-IDF).

    Final score = alpha * BM25_norm + (1 - alpha) * cosine_norm
    """

    name = "hybrid"

    def __init__(self, alpha: float = 0.5) -> None:
        self._alpha = alpha
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = None
        self._bm25: BM25Okapi | None = None
        self._metadata: list[dict[str, Any]] = []

    def fit(self, documents: list[str], metadata: list[dict[str, Any]]) -> None:
        self._metadata = metadata
        self._matrix = self._vectorizer.fit_transform(documents)
        tokenized = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._matrix is None or self._bm25 is None:
            raise RuntimeError("HybridRetriever must be fitted before search.")

        bm25_scores = np.array(self._bm25.get_scores(_tokenize(query)))
        query_vec = self._vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vec, self._matrix).flatten()

        combined = (
            self._alpha * _min_max_normalize(bm25_scores)
            + (1.0 - self._alpha) * _min_max_normalize(cosine_scores)
        )
        ranked_indices = combined.argsort()[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            record = self._metadata[idx]
            results.append(
                {
                    "rank": rank,
                    "algorithm": self.name,
                    "score": round(float(combined[idx]), 4),
                    "bm25_component": round(float(bm25_scores[idx]), 4),
                    "cosine_component": round(float(cosine_scores[idx]), 4),
                    "product": record["product"],
                    "question": record["question"],
                    "answer": record["answer"],
                }
            )
        return results
