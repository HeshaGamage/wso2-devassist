import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from src.retriever import ask, retrieve

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="WSO2 DevAssist",
    page_icon="🔷",
    layout="wide"
)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🔷 WSO2 DevAssist")
st.sidebar.caption("AI-powered assistant for WSO2 API Manager")
st.sidebar.divider()

st.sidebar.subheader("About")
st.sidebar.write(
    "This assistant uses RAG over the official WSO2 API Manager "
    "documentation to answer your questions accurately with source citations."
)

st.sidebar.divider()
st.sidebar.subheader("Try asking:")
example_questions = [
    "What is WSO2 API Manager?",
    "How do I secure an API with OAuth2?",
    "What is the MCP Gateway?",
    "How do I create a REST API?",
    "What are throttling tiers?",
    "How does the AI Gateway work?"
]

for q in example_questions:
    if st.sidebar.button(q, use_container_width=True, key=f"example_{q}"):
        st.session_state['prefill'] = q

st.sidebar.divider()
st.sidebar.caption("Powered by RAG + Llama 3 + ChromaDB")
st.sidebar.caption("Knowledge base: WSO2 APIM official docs")

# ── Initialize session state ─────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'prefill' not in st.session_state:
    st.session_state.prefill = ""

# ── Main area ────────────────────────────────────────────────
st.title("🔷 WSO2 API Manager Assistant")
st.caption(
    "Ask any technical question about WSO2 API Manager. "
    "Answers are grounded in the official documentation."
)

st.divider()

# ── Chat history ─────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Ask me anything about WSO2 API Manager. "
            "Try one of the example questions in the sidebar to get started."
        )
    
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            
            # Show sources for assistant messages
            if msg['role'] == 'assistant' and 'sources' in msg:
                with st.expander("📚 Sources used", expanded=False):
                    for source in msg['sources']:
                        st.write(f"• {source}")
                    st.caption(
                        f"Retrieved {msg.get('chunks_used', 0)} documentation chunks"
                    )

# ── Chat input ────────────────────────────────────────────────
prefill_value = st.session_state.get('prefill', '')
if prefill_value:
    st.session_state.prefill = ''

user_input = st.chat_input(
    "Ask a question about WSO2 API Manager...",
)

# Handle prefill from sidebar buttons
active_input = user_input or prefill_value

if active_input:
    # Add user message to history
    st.session_state.messages.append({
        'role': 'user',
        'content': active_input
    })
    
    # Show user message immediately
    with st.chat_message('user'):
        st.markdown(active_input)
    
    # Generate answer
    with st.chat_message('assistant'):
        with st.spinner("Searching documentation..."):
            
            # Check confidence using top chunk distance
            top_chunks = retrieve(active_input, n_results=5)
            top_distance = top_chunks[0]['distance'] if top_chunks else 1.0
            
            # Get full RAG answer
            result = ask(active_input, n_chunks=5)
        
        # Display answer
        st.markdown(result['answer'])
        
        # Confidence indicator
        if top_distance < 0.3:
            confidence_label = "🟢 High confidence"
            confidence_help = "Strong documentation match found"
        elif top_distance < 0.5:
            confidence_label = "🟡 Medium confidence"
            confidence_help = "Moderate match — verify details in official docs"
            st.warning(
                "⚠️ Medium confidence answer. Some details may be incomplete. "
                "Always verify with the [official WSO2 docs](https://apim.docs.wso2.com).",
                icon="⚠️"
            )
        else:
            confidence_label = "🔴 Low confidence"
            confidence_help = "Question may be outside documentation scope"
            st.error(
                "🔴 Low confidence. This topic may not be well covered in the "
                "indexed documentation. Check the "
                "[official WSO2 docs](https://apim.docs.wso2.com) directly.",
            )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.caption(f"{confidence_label}")
        with col2:
            st.caption(f"Best match distance: {top_distance:.3f} — {confidence_help}")
        
        # Sources expander
        with st.expander("📚 Sources used", expanded=False):
            for source in result['sources']:
                st.write(f"• {source}")
            st.caption(
                f"Retrieved {result['chunks_used']} documentation chunks"
            )
    
    # Save assistant message to history
    st.session_state.messages.append({
        'role': 'assistant',
        'content': result['answer'],
        'sources': result['sources'],
        'chunks_used': result['chunks_used']
    })

# ── Clear chat button ─────────────────────────────────────────
if st.session_state.messages:
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()