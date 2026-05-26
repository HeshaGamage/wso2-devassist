"""
Quick test of MCP server functions without a full MCP client.
Tests the underlying functions directly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_server import retrieve_chunks, generate_rag_answer

print("Testing search_wso2_docs tool...")
chunks = retrieve_chunks("MCP gateway model context protocol", n_results=3)
print(f"Retrieved {len(chunks)} chunks")
for c in chunks:
    print(f"  [{c['distance']}] {c['page_title']} > {c['section']}")

print("\nTesting answer_wso2_question tool...")
question = "How does the MCP Gateway transform APIs into AI tools?"
answer = generate_rag_answer(question, chunks)
print(f"Answer preview: {answer[:400]}...")
print("\nAll tools working correctly.")