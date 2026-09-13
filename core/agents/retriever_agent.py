import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from memory.vector_memory import search_memory
from memory.graph_memory import get_facts_about
from retrieval.hybrid_search import load_documents, build_vector_index, build_bm25_index, hybrid_search, rerank_results

class RetrieverAgent:
    """Specialized agent responsible for gathering all relevant context:
    past conversation memory, knowledge graph facts, and document search."""

    def __init__(self):
        try:
            self.documents = load_documents()
            self.doc_vector_index = build_vector_index(self.documents)
            self.doc_bm25, self.doc_texts = build_bm25_index(self.documents)
            self.retrieval_available = True
        except Exception as e:
            print(f"[RetrieverAgent] Document retrieval unavailable: {e}")
            self.retrieval_available = False

    def get_memory_context(self, task: str):
        try:
            return search_memory(task, top_k=2)
        except Exception as e:
            print(f"[RetrieverAgent] Memory search failed: {e}")
            return []

    def get_graph_context(self):
        try:
            return get_facts_about("User")
        except Exception as e:
            print(f"[RetrieverAgent] Knowledge graph unavailable: {e}")
            return []

    def get_document_context(self, task: str):
        if not self.retrieval_available:
            return []
        try:
            results = hybrid_search(task, self.documents, self.doc_vector_index, self.doc_bm25, self.doc_texts)
            return rerank_results(task, results, top_k=2)
        except Exception as e:
            print(f"[RetrieverAgent] Document search failed: {e}")
            return []

    def gather_context(self, task: str) -> str:
        """Combines all context sources into a single enriched context string."""
        doc_context = self.get_document_context(task)
        graph_facts = self.get_graph_context()
        past_memories = self.get_memory_context(task)

        parts = []
        if doc_context:
            parts.append("Relevant document context (primary source for factual answers):\n" + "\n".join(doc_context))
        if graph_facts:
            parts.append("Known facts about the user:\n" + "\n".join(graph_facts))
        if past_memories:
            parts.append("Past conversation memory (background only, lower priority):\n" + "\n".join(past_memories))

        return "\n\n".join(parts) if parts else "No relevant context available."