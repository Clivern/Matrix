from sentence_transformers import SentenceTransformer
import numpy as np

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


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