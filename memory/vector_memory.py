from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Connect to local Qdrant (running via Docker)
client = QdrantClient(url="http://localhost:6333")

# Small, fast embedding model (converts text -> vectors)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "agent_memory"

def init_memory():
    """Create the collection if it doesn't exist."""
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")

def store_memory(text: str):
    """Save a piece of text (e.g. task+output) into vector memory."""
    vector = embedder.encode(text).tolist()
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=vector, payload={"text": text})]
    )
    return point_id

def search_memory(query: str, top_k: int = 3):
    """Find the most similar past memories to a query."""
    query_vector = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points
    return [r.payload["text"] for r in results]

if __name__ == "__main__":
    init_memory()
    store_memory("The user asked about AI agents and their definition.")
    store_memory("The user is building a self-learning multi-agent system.")
    
    print("\nSearching for: 'What is the user building?'")
    results = search_memory("What is the user building?")
    for r in results:
        print("-", r)