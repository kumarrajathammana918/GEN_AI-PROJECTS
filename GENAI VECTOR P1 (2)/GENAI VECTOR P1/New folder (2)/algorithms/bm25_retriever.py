from typing import Any

from rank_bm25 import BM25Okapi

from .base_retriever import BaseRetriever


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever(BaseRetriever):
    """Retrieve Q&A pairs using BM25 lexical ranking."""

    name = "bm25"

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._metadata: list[dict[str, Any]] = []

    def fit(self, documents: list[str], metadata: list[dict[str, Any]]) -> None:
        self._metadata = metadata
        tokenized = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever must be fitted before search.")

        scores = self._bm25.get_scores(_tokenize(query))
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
