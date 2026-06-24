from typing import Any

import hnswlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base_retriever import BaseRetriever


class HNSWRetriever(BaseRetriever):
    """Retrieve Q&A pairs using HNSW approximate nearest neighbor search on TF-IDF vectors."""

    name = "hnsw"

    def __init__(self, ef_construction: int = 200, m: int = 16) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._index: hnswlib.Index | None = None
        self._metadata: list[dict[str, Any]] = []
        self._ef_construction = ef_construction
        self._m = m

    def fit(self, documents: list[str], metadata: list[dict[str, Any]]) -> None:
        self._metadata = metadata
        matrix = self._vectorizer.fit_transform(documents)
        dense_matrix = normalize(matrix)
        # hnswlib requires a dense numpy array of dtype float32
        if hasattr(dense_matrix, "toarray"):
            dense = dense_matrix.toarray().astype(np.float32)
        else:
            dense = dense_matrix.astype(np.float32)

        num_docs, dim = dense.shape
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=num_docs, ef_construction=self._ef_construction, M=self._m)
        index.add_items(dense, np.arange(num_docs))
        index.set_ef(50)
        self._index = index

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._index is None:
            raise RuntimeError("HNSWRetriever must be fitted before search.")
        query_vec_matrix = normalize(self._vectorizer.transform([query]))
        if hasattr(query_vec_matrix, "toarray"):
            query_vec = query_vec_matrix.toarray().astype(np.float32)
        else:
            query_vec = query_vec_matrix.astype(np.float32)
        self._index.set_ef(max(50, top_k * 2))
        labels, distances = self._index.knn_query(query_vec, k=min(top_k, len(self._metadata)))

        results: list[dict[str, Any]] = []
        for rank, (idx, distance) in enumerate(zip(labels[0], distances[0], strict=True), start=1):
            record = self._metadata[int(idx)]
            similarity = round(1.0 - float(distance), 4)
            results.append(
                {
                    "rank": rank,
                    "algorithm": self.name,
                    "score": similarity,
                    "score_label": "cosine similarity via HNSW",
                    "product": record["product"],
                    "question": record["question"],
                    "answer": record["answer"],
                }
            )
        return results
