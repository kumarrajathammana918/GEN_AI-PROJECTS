# Amazon QA Chatbot

A retrieval-based Q&A chatbot for Amazon product questions. Users log in, ask questions, and compare results from five search algorithms.

## Algorithms (separate files)

| File | Algorithm |
|------|-----------|
| `algorithms/cosine_retriever.py` | TF-IDF + Cosine Similarity |
| `algorithms/euclidean_retriever.py` | TF-IDF + Euclidean Distance |
| `algorithms/bm25_retriever.py` | BM25 lexical ranking |
| `algorithms/hnsw_retriever.py` | HNSW approximate nearest neighbors |
| `algorithms/hybrid_retriever.py` | Hybrid BM25 + Cosine fusion |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Login credentials

| Username | Password |
|----------|----------|
| admin    | admin123 |
| demo     | demo123  |
| analyst  | analyst123 |

## Usage

1. Log in with demo credentials.
2. Enter a product question (e.g. "Does Echo Dot work with Spotify?").
3. Choose **Compare all algorithms** to see side-by-side results, or pick a single algorithm.
4. Adjust the number of results with the slider.

## Project structure

```
├── app.py                      # Streamlit UI with login
├── auth.py                     # Authentication
├── retriever_manager.py        # Loads data and runs retrievers
├── algorithms/
│   ├── cosine_retriever.py
│   ├── euclidean_retriever.py
│   ├── bm25_retriever.py
│   ├── hnsw_retriever.py
│   └── hybrid_retriever.py
├── data/
│   └── amazon_qa_sample.json   # Sample Amazon Q&A dataset
└── utils/
    └── data_loader.py
```
