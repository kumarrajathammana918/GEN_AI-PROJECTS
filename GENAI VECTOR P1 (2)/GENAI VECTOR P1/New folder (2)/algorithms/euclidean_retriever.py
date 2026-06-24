from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_retriever import BaseRetriever


class EuclideanRetriever(BaseRetriever):
    """Retrieve Q&A pairs using TF-IDF vectors and Euclidean distance."""

    name = "euclidean"

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = None
        self._metadata: list[dict[str, Any]] = []

    def fit(self, documents: list[str], metadata: list[dict[str, Any]]) -> None:
        self._metadata = metadata
        self._matrix = self._vectorizer.fit_transform(documents).toarray()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._matrix is None:
            raise RuntimeError("EuclideanRetriever must be fitted before search.")

        query_vec = self._vectorizer.transform([query]).toarray()[0]
        distances = np.linalg.norm(self._matrix - query_vec, axis=1)
        ranked_indices = distances.argsort()[:top_k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            record = self._metadata[idx]
            distance = float(distances[idx])
            results.append(
                {
                    "rank": rank,
                    "algorithm": self.name,
                    "score": round(distance, 4),
                    "score_label": "distance (lower is better)",
                    "product": record["product"],
                    "question": record["question"],
                    "answer": record["answer"],
                }
            )
        return results
