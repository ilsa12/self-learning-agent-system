# Self-Learning Multi-Agent Ecosystem with Graph Memory

An agentic AI system that combines self-correcting reasoning (Reflexion), dual memory (vector + knowledge graph), and enterprise search — built as a hands-on exploration of production-grade agentic AI architecture.

## Current Progress

- ✅ **Phase 1: Core Agentic Engine** — Self-correcting agent loop using Reflexion. The agent generates a response, evaluates its own output quality, and retries with feedback if needed.
- ✅ **Phase 2: Dual-Memory System**
  - **Vector Memory (Qdrant)** — stores past interactions as embeddings for semantic similarity search.
  - **Knowledge Graph (Neo4j)** — stores structured facts and relationships between entities (e.g. User → USES → LangGraph).
- ⏳ **Phase 3: Enterprise Search (Hybrid RAG)** — in progress
- ⏳ **Phase 4: Production Deployment** — planned

## Tech Stack

- **Language/Framework**: Python, LangGraph, FastAPI, Pydantic
- **LLM**: Ollama (Llama 3.1) — fully local, no paid API
- **Vector DB**: Qdrant (via Docker)
- **Knowledge Graph**: Neo4j (via Docker)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)

## Architecture

User Task → FastAPI → Agent Core
↓
┌───────────┴───────────┐
↓ ↓
Vector Memory Knowledge Graph
(Qdrant) (Neo4j)
↓ ↓
└───────────┬───────────┘
↓
Enriched Context
↓
LLM Generation
↓
Reflexion (Self-Evaluation)
↓
Retry if BAD → Final Output


## How to Run

1. Install [Ollama](https://ollama.com) and pull a model: `ollama pull llama3.1`
2. Run Qdrant and Neo4j via Docker (see `docker-compose.yml`)
3. Install dependencies: `pip install -r requirements.txt`
4. Start the API: `uvicorn api.main:app --reload`
5. Visit `http://127.0.0.1:8000/docs` to test the agent

## Project Structure

agentic-project/
├── core/ # Agent logic + Reflexion loop
├── memory/ # Vector memory (Qdrant) + Knowledge graph (Neo4j)
├── retrieval/ # Enterprise RAG (Phase 3, in progress)
├── api/ # FastAPI application
└── requirements.txt