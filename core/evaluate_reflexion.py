import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.agents.retriever_agent import RetrieverAgent
from core.agents.generator_agent import GeneratorAgent
from core.agents.critic_agent import CriticAgent

# Test queries with expected keywords in a correct answer
test_cases = [
    ("How much does the product cost?", "$49"),
    ("How many sick leave days do employees get?", "12"),
    ("Can I work remotely?", "remote"),
    ("What is the API rate limit?", "1,000"),
    ("When are performance reviews conducted?", "June"),
    ("How much annual leave do I get?", "20"),
    ("What integrations are supported?", "Slack"),
    ("How long is parental leave?", "12 weeks"),
    ("What is the refund policy?", "refund policy"),
    ("Who is the CEO of the company?", "CEO"),
]

def run_reflexion_evaluation():
    print("Initializing agents...\n")
    retriever = RetrieverAgent()
    generator = GeneratorAgent()
    critic = CriticAgent()

    first_attempt_correct = 0
    final_attempt_correct = 0
    total = len(test_cases)
    retries_triggered = 0

    for task, expected_keyword in test_cases:
        context = retriever.gather_context(task)

        # First attempt (no reflexion)
        first_output = generator.generate(task, context)
        first_correct = expected_keyword.lower() in first_output.lower()

        # Full reflexion loop
        output = first_output
        attempt = 0
        max_retries = 2
        retried = False
        while attempt < max_retries:
            verdict = critic.evaluate(task, output)
            if critic.is_good(verdict):
                break
            retried = True
            output = generator.regenerate_with_feedback(task, context, output, verdict)
            attempt += 1

        final_correct = expected_keyword.lower() in output.lower()

        if retried:
            retries_triggered += 1
        if first_correct:
            first_attempt_correct += 1
        if final_correct:
            final_attempt_correct += 1

        status_change = ""
        if not first_correct and final_correct:
            status_change = " <-- IMPROVED by reflexion"
        elif first_correct and not final_correct:
            status_change = " <-- WORSENED by reflexion"

        print(f"Task: {task}")
        print(f"  First attempt correct: {first_correct} | Final attempt correct: {final_correct}{status_change}")
        print()

    print("=" * 60)
    print(f"First-attempt accuracy:  {first_attempt_correct}/{total} ({first_attempt_correct/total*100:.1f}%)")
    print(f"Final-attempt accuracy:  {final_attempt_correct}/{total} ({final_attempt_correct/total*100:.1f}%)")
    print(f"Retries triggered:       {retries_triggered}/{total}")
    print("=" * 60)

if __name__ == "__main__":
    run_reflexion_evaluation()