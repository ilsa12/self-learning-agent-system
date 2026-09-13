from langchain_ollama import ChatOllama

class CriticAgent:
    """Specialized agent responsible for evaluating the quality of generated
    outputs (Reflexion) and deciding whether a retry is needed."""

    def __init__(self, model="llama3.1"):
        self.llm = ChatOllama(model=model)

    def evaluate(self, task: str, output: str) -> str:
        eval_prompt = f"""Task: {task}
Output: {output}

Evaluate ONLY whether the output directly and correctly answers the task.
Minor omissions of unrelated background context are fine and should NOT be marked as bad.
Reply with only "GOOD" or "BAD" followed by a one-line reason.
Only say BAD if the output fails to answer the actual question asked, contains factual errors, or is confusing/contradictory."""
        result = self.llm.invoke(eval_prompt)
        return result.content

    def is_good(self, verdict: str) -> bool:
        return verdict.strip().upper().startswith("GOOD")