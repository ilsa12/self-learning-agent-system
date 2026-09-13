from langchain_ollama import ChatOllama

class GeneratorAgent:
    """Specialized agent responsible for generating responses using the LLM,
    given a task and enriched context."""

    def __init__(self, model="llama3.1"):
        self.llm = ChatOllama(model=model)

    def generate(self, task: str, context: str) -> str:
        prompt = f"{context}\n\nCurrent task: {task}\n\nInstructions: Answer the task directly using the document context above if it's relevant. Be concise and direct."
        response = self.llm.invoke(prompt)
        return response.content

    def regenerate_with_feedback(self, task: str, context: str, previous_output: str, verdict: str) -> str:
        prompt = f"""{context}

Current task: {task}

Your previous attempt was: {previous_output}
It was judged as: {verdict}

Focus specifically on directly answering the original task using the most relevant piece of context above (prioritize the document context if present). Do not overthink — give a direct, concise answer."""
        response = self.llm.invoke(prompt)
        return response.content