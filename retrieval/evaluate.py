from hybrid_search import load_documents, build_vector_index, build_bm25_index, hybrid_search, rerank_results

# Test set: query + keyword that MUST appear in the correct answer
test_queries = [
    ("How much does the product cost?", "$49"),
    ("How many sick leave days do employees get?", "12 days"),
    ("Can I work from home?", "remotely"),
    ("How do I reset my password?", "Reset Password"),
    ("What is the API rate limit?", "1,000 requests"),
    ("When are performance reviews conducted?", "June and December"),
    ("How much annual leave do I get?", "20 days"),
    ("What integrations are supported?", "Slack"),
    ("How long is parental leave?", "12 weeks"),
    ("When was the company founded?", "2019"),
]

def run_evaluation():
    print("Loading documents and building indexes...\n")
    documents = load_documents()
    vector_index = build_vector_index(documents)
    bm25, texts = build_bm25_index(documents)

    correct = 0
    total = len(test_queries)

    for query, expected_keyword in test_queries:
        results = hybrid_search(query, documents, vector_index, bm25, texts, top_k=3)
        reranked = rerank_results(query, results, top_k=1)

        top_result = reranked[0] if reranked else ""
        is_correct = expected_keyword.lower() in top_result.lower()

        status = "PASS" if is_correct else "FAIL"
        print(f"[{status}] Query: {query}")
        if not is_correct:
            print(f"   Expected keyword: '{expected_keyword}'")
            print(f"   Got: {top_result[:100]}...")

        if is_correct:
            correct += 1

    accuracy = (correct / total) * 100
    print(f"\n{'='*50}")
    print(f"Retrieval Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_evaluation()