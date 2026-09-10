from langchain_ollama import ChatOllama
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from memory.vector_memory import init_memory, store_memory, search_memory
from memory.graph_memory import add_fact, get_facts_about
from retrieval.hybrid_search import load_documents, build_vector_index, build_bm25_index, hybrid_search, rerank_results

llm = ChatOllama(model="llama3.1")

# Try to initialize memory - don't crash if Qdrant is down
try:
    init_memory()
    MEMORY_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Vector memory unavailable: {e}")
    MEMORY_AVAILABLE = False

# Try to load documents for retrieval
try:
    documents = load_documents()
    doc_vector_index = build_vector_index(documents)
    doc_bm25, doc_texts = build_bm25_index(documents)
    RETRIEVAL_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Document retrieval unavailable: {e}")
    RETRIEVAL_AVAILABLE = False

def generate(task: str) -> str:
    response = llm.invoke(task)
    return response.content

def evaluate(task: str, output: str) -> str:
    eval_prompt = f"""Task: {task}
Output: {output}

Is this output good quality and correct? Reply with only "GOOD" or "BAD" followed by a one-line reason."""
    result = llm.invoke(eval_prompt)
    return result.content

def safe_search_memory(task):
    if not MEMORY_AVAILABLE:
        return []
    try:
        return search_memory(task, top_k=2)
    except Exception as e:
        print(f"[WARNING] Memory search failed: {e}")
        return []

def safe_graph_facts():
    try:
        return get_facts_about("User")
    except Exception as e:
        print(f"[WARNING] Knowledge graph unavailable: {e}")
        return []

def safe_doc_search(task):
    if not RETRIEVAL_AVAILABLE:
        return []
    try:
        results = hybrid_search(task, documents, doc_vector_index, doc_bm25, doc_texts)
        return rerank_results(task, results, top_k=2)
    except Exception as e:
        print(f"[WARNING] Document search failed: {e}")
        return []

def run_agent(task: str, max_retries: int = 2) -> str:
    past_memories = safe_search_memory(task)
    graph_facts = safe_graph_facts()
    doc_context = safe_doc_search(task)

    context_parts = []
    if past_memories:
        context_parts.append("Past conversation memory:\n" + "\n".join(past_memories))
    if graph_facts:
        context_parts.append("Known facts about the user:\n" + "\n".join(graph_facts))
    if doc_context:
        context_parts.append("Relevant document context:\n" + "\n".join(doc_context))

    context = "\n\n".join(context_parts) if context_parts else "No relevant context available."
    enriched_task = f"{context}\n\nCurrent task: {task}"

    attempt = 0
    output = generate(enriched_task)

    while attempt < max_retries:
        verdict = evaluate(task, output)
        print(f"\n[Reflexion check - attempt {attempt+1}]: {verdict}\n")

        if verdict.strip().upper().startswith("GOOD"):
            break

        retry_prompt = f"{enriched_task}\n\nYour previous attempt was: {output}\nIt was judged as: {verdict}\nPlease improve and try again."
        output = generate(retry_prompt)
        attempt += 1

    if MEMORY_AVAILABLE:
        try:
            store_memory(f"Task: {task}\nOutput: {output}")
        except Exception as e:
            print(f"[WARNING] Could not store memory: {e}")

    return output

if __name__ == "__main__":
    result = run_agent("How much does the product cost?")
    print("FINAL OUTPUT:\n", result)