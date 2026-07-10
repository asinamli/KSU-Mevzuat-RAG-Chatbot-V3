from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "mevzuat_rag"

res = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=1
)

if res[0]:
    print("Veritabanındaki Veri Örneği:")
    print(res[0][0].payload)
else:
    print("Veritabanı boş!")