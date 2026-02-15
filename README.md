# RAG Playground

A hands-on experimentation repository featuring foundational RAG (Retrieval-Augmented Generation) concepts to production-style agentic RAG systems —> culminating in a voice-enabled insurance assistant POC.

## Project Structure

```
rag_playground/
│
├── data_ingestion/               # Step 1: Loading & parsing documents
│   ├── ingestion1.ipynb          # Multi-format data loading (PDF, DOCX, TXT)
│   └── data/
│
├── embeddings_vectorDB/          # Step 2: Embeddings & vector stores
│   ├── embeddings.ipynb          # Embedding techniques
│   ├── rag_chromdb.ipynb         # RAG with ChromaDB
│   ├── rag_faiss.ipynb           # RAG with FAISS
│   └── pinecone_test.ipynb       # Pinecone cloud vector DB
│
├── hybrid_search_strategies/     # Step 3: Combining search methods
│   ├── hybrid_search.ipynb       # BM25 + vector search fusion
│   └── re_ranking.ipynb          # Re-ranking retrieved documents
│
├── query_enhancement/            # Step 4: Query optimization
│   ├── query_decomposition.ipynb # Breaking complex queries into sub-queries
│   └── hyDE.ipynb                # Hypothetical Document Embeddings
│
├── agentic_rag/                  # Step 5: Agentic RAG patterns
│   ├── reACT_agentic_rag.ipynb   # ReACT pattern with multi-tool orchestration
│   └── autonomous_rag.ipynb      # Autonomous RAG with planning & reflection
│
├── insurance_agent_poc/          # Step 6: Real-world POC
│   ├── app.py                    # Streamlit chat app (text)
│   ├── appV2.py                  # Streamlit chat app (voice-enabled)
│   └── policies/                 # Sample insurance policy documents
│
├── graph_rag/                    # (WIP) Graph-based RAG
├── advanced_rag/                 # (WIP) Caching patterns
└── multimodal_rag/               # (WIP) Multi-modal RAG
```

## Learning Path

| Stage | Topic | Key Concepts |
|-------|-------|-------------|
| 1 | **Data Ingestion** | PDF/DOCX/TXT parsing, document loaders, text splitting |
| 2 | **Embeddings & Vector DBs** | Sentence embeddings, ChromaDB, FAISS, Pinecone |
| 3 | **Hybrid Search** | BM25 + vector search, reciprocal rank fusion, re-ranking |
| 4 | **Query Enhancement** | Query decomposition, HyDE (Hypothetical Document Embeddings) |
| 5 | **Agentic RAG** | ReACT pattern, autonomous agents, tool selection, context grading |
| 6 | **Insurance Agent POC** | Multi-source orchestration, voice interface, production patterns |

## Insurance Agent POC

The capstone project is a conversational health insurance assistant built with **LangGraph** that orchestrates multiple data sources to answer member queries.

### Architecture

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Authenticate │────▶│ Policy Retrieval  │────▶│ Provider Check  │
│  (MongoDB)   │     │   (FAISS RAG)    │     │ (Network DBs)   │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │    Synthesize    │
                                              │  (Gemini LLM)   │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                                   Response
```

### Features

- **Multi-source retrieval** — Combines member database, policy documents (via RAG), and provider network data
- **LangGraph orchestration** — State-based workflow with conditional routing
- **Voice interface (V2)** — Speech-to-text and text-to-speech via ElevenLabs
- **Waiting period validation** — Checks policy dates against pre-existing conditions
- **Network hospital lookup** — Verifies cashless claim eligibility

### Running the POC

```bash
cd insurance_agent_poc

# Text-based chat
streamlit run app.py

# Voice-enabled chat
streamlit run appV2.py
```

## Agentic RAG Notebooks

### ReACT Agentic RAG
Multi-tool orchestration using the ReACT (Reasoning + Acting) pattern:
- 3 domain-specific vector retriever tools
- Wikipedia, ArXiv, and Tavily web search tools
- Context grading with automatic query rewriting on low relevance

### Autonomous RAG
Self-directed RAG agent with:
- Query decomposition and planning
- Automatic tool selection (retriever vs. web search)
- Reflection mechanism with retry loops

## Tech Stack

| Category | Tools |
|----------|-------|
| **LLM** | Google Gemini 2.5 Flash (via LangChain) |
| **Orchestration** | LangChain, LangGraph |
| **Embeddings** | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Stores** | ChromaDB, FAISS, Pinecone |
| **Search** | BM25 (rank-bm25), Tavily Web Search |
| **Voice** | ElevenLabs (STT + TTS) |
| **UI** | Streamlit |
| **Database** | MongoDB (mongomock for POC) |
| **Data Parsing** | PyPDF, PyMuPDF, Unstructured, python-docx |

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd rag_playground

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
```

| Variable | Required For |
|----------|-------------|
| `GOOGLE_API_KEY` | All notebooks and the insurance POC (Gemini LLM) |
| `PINECONE_API_KEY` | Pinecone vector DB notebook only |
| `TAVILY_API_KEY` | Agentic RAG web search tools |
