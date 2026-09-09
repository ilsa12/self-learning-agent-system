from langchain_ollama import ChatOllama
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from memory.vector_memory import init_memory, store_memory, search_memory
from memory.graph_memory import add_fact, get_facts_about
from retrieval.hybrid_search import load_documents, build_vector_index, build_bm25_index, hybrid_search, rerank_results

llm = ChatOllama(model="llama3.1")
init_memory()

# Load documents once at startup for retrieval
documents = load_documents()
doc_vector_index = build_vector_index(documents)
doc_bm25, doc_texts = build_bm25_index(documents)

def generate(task: str) -> str:
    response = llm.invoke(task)
    return response.content

def evaluate(task: str, output: str) -> str:
    eval_prompt = f"""Task: {task}
Output: {output}

Is this output good quality and correct? Reply with only "GOOD" or "BAD" followed by a one-line reason."""
    result = llm.invoke(eval_prompt)
    return result.content

def run_agent(task: str, max_retries: int = 2) -> str:
    # Step 1: Get context from all sources
    past_memories = search_memory(task, top_k=2)
    graph_facts = get_facts_about("User")
    doc_results = hybrid_search(task, documents, doc_vector_index, doc_bm25, doc_texts)
    doc_context = rerank_results(task, doc_results, top_k=2)

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

    store_memory(f"Task: {task}\nOutput: {output}")
    return output

if __name__ == "__main__":
    result = run_agent("How much does the product cost?")
    print("FINAL OUTPUT:\n", result)