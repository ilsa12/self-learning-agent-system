# Self-Learning Multi-Agent Ecosystem with Graph Memory & Enterprise Search

An agentic AI system built from scratch to explore production-grade agentic architecture: a true multi-agent pipeline with self-correcting reasoning (Reflexion), dual memory (vector + knowledge graph), and enterprise hybrid search (RAG) — fully local, no paid APIs.

## What It Does

Ask the system a question (e.g. "How much does the product cost?") and it:
1. Retrieves relevant context from past conversations, a knowledge graph, and internal documents
2. Generates a response using a local LLM
3. Critiques its own output and retries if the answer is inadequate
4. Stores the interaction back into memory for future use

## Architecture: 3 Specialized Agents + Orchestrator

                    Orchestrator
                         |
    ┌────────────────────┼────────────────────┐
    ↓                    ↓                     ↓
    Retriever Agent Generator Agent Critic Agent
| | |
┌─────┼─────┐ | |
↓ ↓ ↓ | |
Vector Graph Documents | |
Memory (Neo4j) (Hybrid RAG) | |
| | | | |
└─────┴─────┴────────────────┘ |
↓ |
Enriched Context ──────────────────→ LLM Response ─┘
↓
GOOD → Final Output
BAD → Retry with feedback
↓
Stored in Vector Memory


- **Retriever Agent** — gathers context from vector memory (Qdrant), knowledge graph (Neo4j), and hybrid document search
- **Generator Agent** — produces responses using the local LLM (Ollama / Llama 3.1), and regenerates with feedback when needed
- **Critic Agent** — evaluates output quality (Reflexion) and decides whether a retry is needed
- **Orchestrator** — coordinates the full pipeline end-to-end

## Features

- ✅ **Self-Correcting Reflexion Loop** — the Critic Agent evaluates outputs and triggers retries with feedback when quality is inadequate
- ✅ **Vector Memory (Qdrant)** — semantic search over past conversations
- ✅ **Knowledge Graph (Neo4j)** — structured facts and relationships (e.g. `User -[IS_BUILDING]-> Multi_Agent_System`)
- ✅ **Hybrid RAG** — combines BM25 keyword search + dense vector search, with cross-encoder reranking (FlashRank) for high-precision retrieval over real PDF/text documents
- ✅ **Document Chunking** — documents are split into focused chunks for precise retrieval
- ✅ **Interactive Dashboard** — Streamlit UI showing live chat, knowledge graph facts, and system status
- ✅ **Error Handling** — the system degrades gracefully (with warnings) if Qdrant or Neo4j are unavailable, instead of crashing
- ✅ **Evaluation Suite** — automated scripts measuring retrieval accuracy and reflexion effectiveness

## Evaluation Results

- **Retrieval accuracy**: 90% on a 10-query evaluation set (hybrid search + reranking)
- **Reflexion evaluation**: 100% first-attempt and final-attempt accuracy on test queries — indicating the retrieval pipeline is precise enough that the Critic Agent rarely needs to intervene, acting as a reliability safety-net rather than a necessity
- **Learning note**: keyword-based evaluation has limitations (e.g. a correct "no answer found" response can accidentally match an expected keyword) — a good example of why evaluation methodology itself needs scrutiny

## Tech Stack

- **Language/Framework**: Python, LangGraph-style multi-agent orchestration, FastAPI, Pydantic
- **LLM**: Ollama (Llama 3.1) — fully local, no paid API
- **Vector DB**: Qdrant (via Docker)
- **Knowledge Graph**: Neo4j (via Docker)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2), BAAI/bge-small-en-v1.5
- **Retrieval**: LlamaIndex, rank-bm25, FlashRank (cross-encoder reranking)
- **UI**: Streamlit
- **Containerization**: Docker

## Project Structure
agentic-project/
├── core/
│ ├── agents/
│ │ ├── retriever_agent.py # Gathers context from memory, graph, documents
│ │ ├── generator_agent.py # Generates responses via LLM
│ │ └── critic_agent.py # Evaluates output quality (Reflexion)
│ ├── orchestrator.py # Coordinates the multi-agent pipeline
│ ├── agent.py # Original single-agent implementation (v1)
│ └── evaluate_reflexion.py # Compares first-attempt vs. final-attempt accuracy
├── memory/
│ ├── vector_memory.py # Qdrant vector memory
│ └── graph_memory.py # Neo4j knowledge graph
├── retrieval/
│ ├── hybrid_search.py # BM25 + vector search + reranking
│ ├── evaluate.py # Retrieval accuracy evaluation
│ └── documents/ # Sample documents (PDF + text)
├── api/
│ └── main.py # FastAPI application
├── ui/
│ └── dashboard.py # Streamlit interactive dashboard
└── requirements.txt

## How to Run

1. Install [Ollama](https://ollama.com) and pull a model:
ollama pull llama3.1

2. Start Qdrant and Neo4j via Docker:
docker run -d --name qdrant-memory -p 6333:6333 qdrant/qdrant
docker run -d --name neo4j-graph -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:latest

3. Install dependencies:
pip install -r requirements.txt

4. Run the multi-agent system directly:
python core/orchestrator.py

5. Or launch the interactive dashboard:
streamlit run ui/dashboard.py

6. Or use the API:

uvicorn api.main:app --reload

   Visit `http://127.0.0.1:8000/docs`

## Scalability Considerations

This is a local prototype, but the architecture maps cleanly to production scale:
- **Qdrant** → managed Qdrant Cloud cluster with sharding for larger document sets
- **Neo4j** → Neo4j Aura for managed, scalable graph storage
- **Batch document ingestion** → async pipelines instead of loading all documents at startup
- **LLM** → swap Ollama for a hosted inference endpoint (vLLM, Bedrock, etc.) for concurrent users
- **Caching** → cache frequent queries/embeddings to reduce redundant LLM and retrieval calls

## Known Limitations

- Tested at small scale (a handful of documents) — not yet load-tested at enterprise scale
- Reflexion (Critic Agent) rarely triggers on the current evaluation set since retrieval is already highly precise
- Keyword-based evaluation can produce false positives; a more robust evaluation (e.g. LLM-as-judge or human review) would strengthen the evaluation suite

## Roadmap

- [ ] Docker Compose for one-command startup of all services
- [ ] LLM-as-judge evaluation instead of keyword matching
- [ ] Expand document set to test retrieval at larger scale