from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """Common interface for all retrieval algorithms."""

    name: str = "base"

    @abstractmethod
    def fit(self, documents: list[str], metadata: list[dict[str, Any]]) -> None:
        """Index documents for search."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Return top-k matches with scores.

        Each result dict contains: product, question, answer, score, rank, algorithm.
        """
