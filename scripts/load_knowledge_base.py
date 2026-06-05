import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "knowledge_base"

def create_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    print("Collection created successfully")

def split_into_chunks(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks

def load_documents():
    knowledge_base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "knowledge_base"
    )

    documents = []
    for filename in os.listdir(knowledge_base_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(knowledge_base_path, filename)
            with open(filepath, "r") as f:
                text = f.read()
            documents.append({
                "filename": filename,
                "text": text,
                "category": filename.split("_")[0]
            })

    print(f"Loaded {len(documents)} documents")
    return documents

def store_in_qdrant(documents):
    points = []
    point_id = 0

    for doc in documents:
        chunks = split_into_chunks(doc["text"])

        for chunk in chunks:
            vector = model.encode(chunk).tolist()

            points.append({
                "id": point_id,
                "vector": vector,
                "payload": {
                    "text": chunk,
                    "filename": doc["filename"],
                    "category": doc["category"]
                }
            })
            point_id += 1

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Stored {len(points)} chunks in Qdrant")

if __name__ == "__main__":
    print("Starting knowledge base ingestion...")
    create_collection()
    documents = load_documents()
    store_in_qdrant(documents)
    print("Knowledge base is ready.")