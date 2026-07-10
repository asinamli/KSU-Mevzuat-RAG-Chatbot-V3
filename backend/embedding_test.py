from sentence_transformers import SentenceTransformer
from file_reader import read_all_documents
from chunker import chunk_mevzuat

model = SentenceTransformer("Ytu-ce-cosmos/turkish-e5-large")

def get_embeddings():
    documents = read_all_documents()
    
    chunks = chunk_mevzuat(documents)

    texts = [f"passage: {c['text']}" for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return chunks, embeddings

if __name__ == "__main__":
    chunks, embeddings = get_embeddings()
    print("Toplam chunk:", len(chunks))
    if len(embeddings) > 0:
        print("Embedding boyutu:", embeddings[0].shape)
        print("Örnek Kaynak:", chunks[0]['source']) 