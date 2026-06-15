import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY", None)
)
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "knowledge_base"

def create_collection():
    # Delete existing collection if it exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    print("Collection created successfully")

def split_into_sections(text, filename):
    """
    Split document into sections based on numbered headings.
    Each numbered section becomes one chunk.
    Preserves the document title and section heading with the content.
    """
    lines = text.strip().split("\n")
    
    # Extract document title (first non-empty line)
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
        
        # Detect numbered section headings like "1.", "2.", "1. HEADING"
        is_section_heading = False
        for i in range(1, 20):
            if stripped.startswith(f"{i}.") and len(stripped) > 2:
                is_section_heading = True
                break
        
        if is_section_heading:
            # Save previous section if it exists
            if current_section_lines:
                chunk_text = f"{title}\n{current_heading}\n" + " ".join(current_section_lines)
                chunks.append(chunk_text.strip())
            
            # Start new section
            current_heading = stripped
            current_section_lines = []
        else:
            # Skip the title line since we already captured it
            if stripped != title:
                current_section_lines.append(stripped)
    
    # Save the last section
    if current_section_lines:
        chunk_text = f"{title}\n{current_heading}\n" + " ".join(current_section_lines)
        chunks.append(chunk_text.strip())
    
    # If no sections were found treat whole document as one chunk
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
                continue  # skip very short chunks

            vector = model.encode(chunk).tolist()
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