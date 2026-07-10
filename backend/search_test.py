from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer("Ytu-ce-cosmos/turkish-e5-large")
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "mevzuat_rag"

def search(query, top_k=5):
    query_embedding = model.encode(
        f"query: {query}",
        normalize_embeddings=True
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k
    )

    for i, p in enumerate(results.points):
        print(f"\n--- Sonuç {i+1} | score={p.score:.4f} ---")
        print(p.payload["madde"])
        print(p.payload["text"][:500])

if __name__ == "__main__":
    soru = input("Sorunu yaz: ")
    search(soru)
