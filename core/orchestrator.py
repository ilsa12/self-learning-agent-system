import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.agents.retriever_agent import RetrieverAgent
from core.agents.generator_agent import GeneratorAgent
from core.agents.critic_agent import CriticAgent
from memory.vector_memory import init_memory, store_memory

class Orchestrator:
    """Coordinates the multi-agent workflow:
    Retriever -> Generator -> Critic -> (retry if needed) -> Store memory."""

    def __init__(self):
        try:
            init_memory()
            self.memory_available = True
        except Exception as e:
            print(f"[Orchestrator] Vector memory unavailable: {e}")
            self.memory_available = False

        print("[Orchestrator] Initializing agents...")
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent()
        self.critic = CriticAgent()
        print("[Orchestrator] All agents ready.")

    def run(self, task: str, max_retries: int = 2) -> str:
        # Step 1: Retriever Agent gathers context
        print("[Orchestrator] Retriever Agent gathering context...")
        context = self.retriever.gather_context(task)

        # Step 2: Generator Agent creates initial response
        print("[Orchestrator] Generator Agent creating response...")
        output = self.generator.generate(task, context)

        # Step 3: Critic Agent evaluates, Generator retries if needed
        attempt = 0
        while attempt < max_retries:
            verdict = self.critic.evaluate(task, output)
            print(f"[Orchestrator] Critic Agent verdict (attempt {attempt+1}): {verdict}")

            if self.critic.is_good(verdict):
                break

            print("[Orchestrator] Generator Agent regenerating with feedback...")
            output = self.generator.regenerate_with_feedback(task, context, output, verdict)
            attempt += 1

        # Step 4: Store the final interaction into memory
        if self.memory_available:
            try:
                store_memory(f"Task: {task}\nOutput: {output}")
            except Exception as e:
                print(f"[Orchestrator] Could not store memory: {e}")

        return output


if __name__ == "__main__":
    orchestrator = Orchestrator()
    result = orchestrator.run("How much does the product cost?")
    print("\nFINAL OUTPUT:\n", result)