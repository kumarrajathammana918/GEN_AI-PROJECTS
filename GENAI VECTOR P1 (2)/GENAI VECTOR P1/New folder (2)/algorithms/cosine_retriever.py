from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_retriever import BaseRetriever


class CosineRetriever(BaseRetriever):
    """Retrieve Q&A pairs using TF-IDF vectors and cosine similarity."""

    name = "cosine"

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = None
        self._metadata: list[dict[str, Any]] = []

    def fit(self, documents: list[str], metadata: list[dict[str, Any]]) -> None:
        self._metadata = metadata
        self._matrix = self._vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._matrix is None:
            raise RuntimeError("CosineRetriever must be fitted before search.")

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            record = self._metadata[idx]
            results.append(
                {
                    "rank": rank,
                    "algorithm": self.name,
                    "score": round(float(scores[idx]), 4),
                    "product": record["product"],
                    "question": record["question"],
                    "answer": record["answer"],
                }
            )
        return results
