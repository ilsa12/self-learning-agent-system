from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
import os

# Setup: use local embedding model + local LLM (no paid API)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)

DOCS_PATH = os.path.join(os.path.dirname(__file__), "documents")

ranker = Ranker()

def load_documents():
    reader = SimpleDirectoryReader(DOCS_PATH)
    return reader.load_data()

def build_vector_index(documents):
    """Dense/semantic search index."""
    return VectorStoreIndex.from_documents(documents)

def build_bm25_index(documents):
    """Keyword-based search index."""
    texts = [doc.text for doc in documents]
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, texts

def hybrid_search(query: str, documents, vector_index, bm25, texts, top_k=2):
    # Vector search results
    vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
    vector_results = vector_retriever.retrieve(query)
    vector_texts = [r.text for r in vector_results]

    # BM25 keyword search results
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_texts = [texts[i] for i in top_bm25_idx]

    # Combine and deduplicate
    combined = list(dict.fromkeys(vector_texts + bm25_texts))
    return combined

def rerank_results(query: str, results: list, top_k=2):
    """Rerank combined results using a cross-encoder for higher precision."""
    passages = [{"id": i, "text": text} for i, text in enumerate(results)]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)
    return [r["text"] for r in reranked[:top_k]]

if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.\n")

    print("Building indexes...")
    vector_index = build_vector_index(documents)
    bm25, texts = build_bm25_index(documents)

    query = "How much does the product cost?"
    print(f"\nQuery: {query}\n")
    results = hybrid_search(query, documents, vector_index, bm25, texts)

    print("Hybrid search results (before reranking):")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}\n")

    print("\n--- After Reranking ---")
    reranked_results = rerank_results(query, results)
    for i, r in enumerate(reranked_results, 1):
        print(f"{i}. {r}\n")