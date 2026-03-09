---
title: Understanding Embeddings
date: 2025-09-10 00:00:00
featured_image: https://images.unsplash.com/photo-1635070041078-e363dbe005cb?q=90&fm=jpg&w=1000&fit=max
excerpt: An embedding is just a way to turn text into a list of numbers. Same idea as coordinates on a map: sentences that mean similar things end up close together, even when they use different words.
keywords: embeddings, sentence-transformers, bge-m3, vector-search, chromadb
---

![](https://images.unsplash.com/photo-1635070041078-e363dbe005cb?q=90&fm=jpg&w=1000&fit=max)

An embedding is just a way to turn text into a list of numbers. Same idea as coordinates on a map: sentences that mean similar things end up close together, even when they use different words.

In a RAG setup you do this in two passes. First, you split your docs into chunks, turn each chunk into numbers, and save those numbers in a database (Qdrant, Chroma, whatever). Later, when someone asks a question, you turn the question into numbers too, find the chunks that are closest, and send only those chunks to the LLM. The LLM does not read your whole library, something else finds the good bits first. Embeddings are what let you search by meaning, not by exact words.

### How matching works?

1. Turn text into vectors (same model, same space).
2. Compare the query vector to each document vector.
3. Highest similarity or lowest distance is your match.

### Setup

Install what we need with [uv](https://docs.astral.sh/uv/):

```zsh
$ uv init
$ uv add numpy sentence-transformers
```

### Turn text into vectors

We use four short docs and one question. The code turns each line of text into a list of numbers using the `BAAI/bge-m3` model from Hugging Face:

```python
from sentence_transformers import SentenceTransformer

DOCUMENTS = [
    "Python is a high-level, general-purpose programming language.",
    "The PyTorch library is widely used for deep learning and neural networks.",
    "ChromaDB is an open-source vector database specialized for AI embeddings.",
    "OpenAI relies on cloud-hosted API architecture for model evaluation.",
]

QUERY = "What tool should I use to store vector data locally?"

model = SentenceTransformer("BAAI/bge-m3")
document_vectors = model.encode(DOCUMENTS, normalize_embeddings=True)
query_vector = model.encode(QUERY, normalize_embeddings=True)
```

`normalize_embeddings=True` scales each list of numbers so they all have the same length.

We use the same model for the docs and the question, so everything ends up in the same kind of space and the scores mean something.

### How close each doc is to the question

Score each doc against the question:
- higher cosine = more similar
- lower L2 distance = closer.

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
```

### Rank all docs at once

```python
def rank_matches(query_vector, document_vectors):
    cosine_scores = document_vectors @ query_vector
    l2_distances = np.linalg.norm(document_vectors - query_vector, axis=1)

    ranked = sorted(
        enumerate(cosine_scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        (idx, float(cosine_scores[idx]), float(l2_distances[idx]))
        for idx, _ in ranked
    ]

results = rank_matches(query_vector, document_vectors)

best_idx = results[0][0]
best_doc = DOCUMENTS[best_idx]

print(f"Query: {QUERY}")
print(f"\nBest match: {best_doc}")
```

Run it and the script prints your question plus the best-matching doc:

```zsh
$ uv run python main.py

Query: What tool should I use to store vector data locally?
Best match: ChromaDB is an open-source vector database specialized for AI embeddings.
```


You can swap in your own documents, embed a whole folder, or plug this into a RAG flow. The model and the compare step stay the same.

[Full source code on GitHub](https://github.com/Clivern/Matrix/tree/main/docs/_code/understanding-embeddings).
