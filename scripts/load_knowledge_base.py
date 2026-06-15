import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from google import genai

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY", None)
)

genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

COLLECTION_NAME = "knowledge_base"

def get_embedding(text):
    result = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def create_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072,
            distance=Distance.COSINE
        )
    )
    print("Collection created successfully")

def split_into_sections(text, filename):
    lines = text.strip().split("\n")

    title = ""
    for line in lines:
        if line.strip():
            title = line.strip()
            break

    chunks = []
    current_section_lines = []
    current_heading = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        is_section_heading = False
        for i in range(1, 20):
            if stripped.startswith(f"{i}.") and len(stripped) > 2:
                is_section_heading = True
                break

        if is_section_heading:
            if current_section_lines:
                chunk_text = f"{title}\n{current_heading}\n" + " ".join(current_section_lines)
                chunks.append(chunk_text.strip())
            current_heading = stripped
            current_section_lines = []
        else:
            if stripped != title:
                current_section_lines.append(stripped)

    if current_section_lines:
        chunk_text = f"{title}\n{current_heading}\n" + " ".join(current_section_lines)
        chunks.append(chunk_text.strip())

    if not chunks:
        chunks.append(text.strip())

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
        chunks = split_into_sections(doc["text"], doc["filename"])
        print(f"{doc['filename']} → {len(chunks)} chunks")

        for chunk in chunks:
            if len(chunk.strip()) < 20:
                continue

            # Retry up to 3 times with delay
            for attempt in range(3):
                try:
                    vector = get_embedding(chunk)
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  Retrying chunk after error: {e}")
                        time.sleep(5)
                    else:
                        raise e

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk,
                    "filename": doc["filename"],
                    "category": doc["category"]
                }
            ))
            point_id += 1
            time.sleep(0.5)  # small delay between chunks to avoid rate limits

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"\nTotal chunks stored in Qdrant: {len(points)}")

if __name__ == "__main__":
    print("Starting knowledge base ingestion...")
    create_collection()
    documents = load_documents()
    store_in_qdrant(documents)
    print("\nKnowledge base is ready.")