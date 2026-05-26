"""
WSO2 DevAssist MCP Server

Exposes WSO2 API Manager documentation search and Q&A
as MCP tools that any compatible AI assistant can use.

Tools:
  - search_wso2_docs: semantic search over WSO2 APIM docs
  - answer_wso2_question: full RAG answer with sources
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── Constants ────────────────────────────────────────────────
CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Lazy-loaded clients ──────────────────────────────────────
_collection = None
_groq_client = None


def get_collection():
    global _collection
    if _collection is None:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(
            name="wso2_apim_docs",
            embedding_function=embedding_fn
        )
    return _collection


def get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq_client


# ── Core functions ───────────────────────────────────────────

def retrieve_chunks(query: str, n_results: int = 5) -> list:
    """Search ChromaDB for relevant documentation chunks."""
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page_title": results["metadatas"][0][i]["page_title"],
            "section": results["metadatas"][0][i]["section"],
            "folder": results["metadatas"][0][i]["folder"],
            "distance": round(results["distances"][0][i], 4)
        })
    return chunks


def generate_rag_answer(question: str, chunks: list) -> str:
    """Generate an answer using retrieved chunks as context."""
    
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"--- Source {i+1}: {chunk['page_title']} > {chunk['section']} ---\n"
            f"{chunk['text']}"
        )
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful technical assistant for WSO2 API Manager.
Answer the developer's question using ONLY the provided documentation excerpts.

RULES:
- Be precise and technical
- Include code examples when the docs contain them
- If docs don't have enough info, say so clearly
- Format code blocks with correct language tags
- Keep answers focused and practical

DOCUMENTATION EXCERPTS:
{context}

DEVELOPER QUESTION:
{question}

ANSWER:"""

    response = get_groq().chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()


# ── MCP Server ───────────────────────────────────────────────

server = Server("wso2-devassist")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Register the tools this MCP server exposes."""
    return [
        types.Tool(
            name="search_wso2_docs",
            description=(
                "Search WSO2 API Manager documentation for relevant content. "
                "Returns the most semantically similar documentation chunks "
                "for a given query. Use this to find specific WSO2 documentation "
                "before answering questions about WSO2 API Manager."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query e.g. 'OAuth2 authentication setup'"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="answer_wso2_question",
            description=(
                "Answer a technical question about WSO2 API Manager using "
                "RAG over the official documentation. Returns a detailed answer "
                "with code examples and source citations. Use this when a developer "
                "asks how to do something specific in WSO2 API Manager."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The technical question about WSO2 API Manager"
                    },
                    "n_chunks": {
                        "type": "integer",
                        "description": "Number of doc chunks to use as context (default 5)",
                        "default": 5
                    }
                },
                "required": ["question"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls from MCP clients."""
    
    if name == "search_wso2_docs":
        query = arguments["query"]
        n_results = arguments.get("n_results", 5)
        n_results = min(n_results, 10)  # cap at 10
        
        chunks = retrieve_chunks(query, n_results)
        
        # Format results as readable text
        output_parts = [f"Search results for: '{query}'\n"]
        for i, chunk in enumerate(chunks):
            output_parts.append(
                f"--- Result {i+1} ---\n"
                f"Page: {chunk['page_title']}\n"
                f"Section: {chunk['section']}\n"
                f"Relevance: {1 - chunk['distance']:.2%}\n"
                f"Content:\n{chunk['text'][:500]}...\n"
            )
        
        return [types.TextContent(
            type="text",
            text="\n".join(output_parts)
        )]
    
    elif name == "answer_wso2_question":
        question = arguments["question"]
        n_chunks = arguments.get("n_chunks", 5)
        
        chunks = retrieve_chunks(question, n_chunks)
        answer = generate_rag_answer(question, chunks)
        
        sources = list({
            f"{c['page_title']} > {c['section']}"
            for c in chunks
        })
        
        output = (
            f"ANSWER:\n{answer}\n\n"
            f"SOURCES:\n" + "\n".join(f"• {s}" for s in sources)
        )
        
        return [types.TextContent(
            type="text",
            text=output
        )]
    
    else:
        return [types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]


# ── Entry point ──────────────────────────────────────────────

async def main():
    print("Starting WSO2 DevAssist MCP Server...", flush=True)
    print(f"ChromaDB path: {CHROMA_PATH}", flush=True)
    print("Tools available: search_wso2_docs, answer_wso2_question", flush=True)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())