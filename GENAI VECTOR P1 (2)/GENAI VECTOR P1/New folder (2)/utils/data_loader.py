import json
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "amazon_qa_sample.json"


def load_qa_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    """Load Amazon Q&A records from JSON."""
    dataset_path = path or DATA_PATH
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def prepare_corpus(records: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """
    Build searchable text and metadata from Q&A records.

    Returns:
        documents: Combined product + question + answer text for indexing
        questions: Original question strings
        metadata: Full record dicts aligned with documents
    """
    documents: list[str] = []
    questions: list[str] = []
    metadata: list[dict[str, Any]] = []

    for record in records:
        doc = (
            f"Product: {record['product']}. "
            f"Question: {record['question']}. "
            f"Answer: {record['answer']}"
        )
        documents.append(doc)
        questions.append(record["question"])
        metadata.append(record)

    return documents, questions, metadata
