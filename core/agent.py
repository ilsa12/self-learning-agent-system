from langchain_ollama import ChatOllama
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from memory.vector_memory import init_memory, store_memory, search_memory
from memory.graph_memory import add_fact, get_facts_about

llm = ChatOllama(model="llama3.1")
init_memory()

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
    # Step 1: Get context from BOTH memory systems
    past_memories = search_memory(task, top_k=2)
    graph_facts = get_facts_about("User")

    context_parts = []
    if past_memories:
        context_parts.append("Past conversation memory:\n" + "\n".join(past_memories))
    if graph_facts:
        context_parts.append("Known facts about the user:\n" + "\n".join(graph_facts))
    
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
    result = run_agent("Based on what you know, suggest one improvement for my project.")
    print("FINAL OUTPUT:\n", result)