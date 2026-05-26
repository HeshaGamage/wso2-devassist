import os
import json
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"

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


def retrieve(query, n_results=5):
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
            "distance": results["distances"][0][i]
        })
    return chunks

def ask(question, n_chunks=5):
    """Full RAG pipeline — retrieve then generate."""
    
    # Primary retrieval
    chunks = retrieve(question, n_results=n_chunks)
    
    # Secondary retrieval with expanded query
    # Helps when the original question is broad
    expanded_query = f"WSO2 API Manager {question}"
    extra_chunks = retrieve(expanded_query, n_results=3)
    
    # Merge and deduplicate by page+section
    seen = set()
    merged = []
    for chunk in chunks + extra_chunks:
        key = f"{chunk['page_title']}>{chunk['section']}"
        if key not in seen:
            seen.add(key)
            merged.append(chunk)
    
    # Sort by distance, take top n_chunks
    merged.sort(key=lambda x: x['distance'])
    final_chunks = merged[:n_chunks]

    context_parts = []
    for i, chunk in enumerate(final_chunks):
        context_parts.append(
            f"--- Source {i+1}: {chunk['page_title']} > {chunk['section']} ---\n"
            f"{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful technical assistant for WSO2 API Manager.
Answer the developer's question using ONLY the provided documentation excerpts.

STRICT RULES:
- Use ONLY information explicitly present in the documentation excerpts
- NEVER invent, fabricate, or assume code examples that are not in the docs
- If the docs contain a code example, reproduce it accurately
- If the docs do NOT contain a code example, say "The documentation does not include a code example for this" and explain conceptually instead
- If the docs don't have enough info to answer fully, say exactly what you do and don't know
- Format code blocks with correct language tags only when code is from the docs
- Keep answers focused and practical

DOCUMENTATION EXCERPTS:
{context}

DEVELOPER QUESTION:
{question}

ANSWER:"""

    response = get_groq().chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        max_tokens=1000,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()
    sources = list({
        f"{c['page_title']} > {c['section']}"
        for c in final_chunks
    })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "chunks_used": len(final_chunks)
    }