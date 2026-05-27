# WSO2 DevAssist

An AI-powered developer assistant for WSO2 API Manager. Ask technical questions in plain English and get accurate answers with source citations — grounded in the official WSO2 documentation.

Built to explore how RAG and MCP servers can make complex enterprise documentation actually usable.

**Live demo:** [link]  
**Demo video:** [link]

---

## What it does

- Answers technical questions about WSO2 API Manager accurately
- Cites the exact documentation sections used in every answer
- Shows a confidence indicator based on how well the docs matched the question
- Warns when answers may be incomplete rather than making things up
- Exposes the knowledge base as an MCP server so any MCP-compatible AI assistant can use it as a tool

---

## How it works

```
WSO2 APIM docs (GitHub) → chunking → embeddings → ChromaDB
                                                       ↓
User question → embed query → find similar chunks → Llama 3 → answer + sources
                                                       ↓
                                              MCP server (for AI clients)
```

3,127 documentation chunks indexed from 350 markdown files across 8 WSO2 API Manager documentation sections.

---

## Stack

- **Embeddings** — all-MiniLM-L6-v2 (sentence-transformers)
- **Vector database** — ChromaDB (cosine similarity)
- **LLM** — Llama 3 via Groq API
- **MCP server** — Python MCP SDK
- **Dashboard** — Streamlit

---

## RAG performance

| Metric | Value |
|--------|-------|
| Documentation chunks | 3,127 |
| Chunks in target size range | 85.7% (200–1200 chars) |
| Retrieval score (keyword coverage) | 100% on test queries |
| Embedding dimension | 384 |
| Distance metric | Cosine |

Confidence thresholds based on cosine distance:
- 🟢 High — distance < 0.3 (strong doc match)
- 🟡 Medium — distance 0.3–0.5 (verify with official docs)
- 🔴 Low — distance > 0.5 (outside documentation scope)

---

## MCP Server

The project exposes two MCP tools that any MCP-compatible AI assistant can use:

**`search_wso2_docs(query, n_results)`**
Semantic search over the WSO2 APIM documentation. Returns the most relevant chunks with relevance scores.

**`answer_wso2_question(question, n_chunks)`**
Full RAG answer with source citations. Returns a detailed technical answer grounded in the official docs.

This maps directly to WSO2's own MCP Gateway product — which transforms APIs into AI-discoverable tools following the Model Context Protocol. This project is itself an example of that pattern.

To connect Claude Desktop to this MCP server, add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "wso2-devassist": {
      "command": "python",
      "args": ["src/mcp_server.py"],
      "env": {
        "GROQ_API_KEY": "your_key_here"
      }
    }
  }
}
```

---

## Setup

### Prerequisites
- Python 3.11+
- Free [Groq API key](https://console.groq.com)

### Installation

```bash
git clone https://github.com/HeshaGamage/wso2-devassist
cd wso2-devassist

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
# Add your Groq API key to .env
```

### Build the knowledge base

```bash
# Clone WSO2 API Manager docs
git clone https://github.com/wso2/docs-apim docs/raw/docs-apim

# Run notebooks in order
jupyter notebook
# 01_data_exploration.ipynb
# 02_chunking.ipynb
# 03_embedding.ipynb   ← builds ChromaDB index
# 04_rag_pipeline.ipynb
```

### Run

```bash
# Streamlit chat UI
streamlit run app.py

# MCP server (for AI clients)
python src/mcp_server.py
```

---

## Project structure

```
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Doc analysis, cleaning strategy
│   ├── 02_chunking.ipynb           # Section-based chunking pipeline
│   ├── 03_embedding.ipynb          # ChromaDB vector store setup
│   └── 04_rag_pipeline.ipynb       # RAG pipeline and evaluation
├── src/
│   ├── retriever.py                # Retrieve + generate functions
│   └── mcp_server.py               # MCP server with two tools
├── docs/
│   ├── raw/                        # WSO2 docs (not committed)
│   └── processed/                  # Chunks JSON
├── chroma_db/                      # Vector store (not committed)
├── app.py                          # Streamlit chat interface
└── requirements.txt
```

---

## Why this project for WSO2

WSO2's newest product — WSO2 Agent Manager — is a control plane for managing MCP servers in enterprise environments. This project builds exactly the kind of MCP server their platform is designed to govern: a domain-specific knowledge tool that AI agents can discover and invoke.

The RAG pipeline also demonstrates the same pattern WSO2's own Choreo platform enables for AI application development — retrieval-augmented generation over enterprise knowledge bases.

---



*Hesha Gamage — [LinkedIn](https://www.linkedin.com/in/heshan-kavishka-655381215/)*