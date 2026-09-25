import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from src.agent.graph import run_agent
from src.evaluation.naive_rag import naive_rag_answer
from src.ingestion.embed_and_store import build_vectorstore
from src.retrieval.hybrid_search import get_hybrid_retriever
from src.retrieval.reranker import Reranker
import src.agent.graph as agent_graph

st.set_page_config(page_title="Agentic RAG — Deep Learning Book", layout="wide")

st.title("📚 Agentic RAG over Dive into Deep Learning")
st.caption("Hybrid retrieval + reranking + self-correcting agent, compared against naive RAG")


@st.cache_resource
def load_vectorstore():
    return build_vectorstore(force_rebuild=False)


@st.cache_resource
def load_retriever_and_reranker():
    """
    Pre-load the hybrid retriever (which builds the BM25 index) and reranker
    ONCE per Streamlit session, then hand them to graph.py's module-level
    globals so run_agent() reuses them instead of rebuilding on every click.
    """
    retriever = get_hybrid_retriever()
    reranker = Reranker()
    agent_graph._retriever = retriever
    agent_graph._reranker = reranker
    return retriever, reranker


with st.spinner("Loading models and indexes (first load only)..."):
    vectorstore = load_vectorstore()
    load_retriever_and_reranker()

query = st.text_input("Ask a question about the book:", placeholder="e.g. What is the vanishing gradient problem?")

col1, col2 = st.columns(2)
run_naive = col1.button("Run Naive RAG")
run_agentic = col2.button("Run Agentic RAG")
run_both = st.button("Run Both (Compare)")

if run_naive or run_both:
    if not query:
        st.warning("Enter a question first.")
    else:
        st.subheader("🔹 Naive RAG")
        with st.spinner("Retrieving and generating..."):
            result = naive_rag_answer(query, vectorstore)
        st.markdown(f"**Answer:**\n\n{result['answer']}")
        with st.expander(f"Retrieved {len(result['retrieved_contexts'])} chunks"):
            for i, ctx in enumerate(result["retrieved_contexts"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.text(ctx[:500])

if run_agentic or run_both:
    if not query:
        st.warning("Enter a question first.")
    else:
        st.subheader("🔸 Agentic RAG")
        with st.spinner("Running retrieve → grade → generate → verify loop..."):
            result = run_agent(query)

        st.markdown(f"**Answer:**\n\n{result['generation']}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Grounded", "✅ Yes" if result["is_grounded"] else "❌ No")
        m2.metric("Retrieval attempts", result["retrieval_attempts"])
        m3.metric("Generation attempts", result["generation_attempts"])

        if result["relevance_scores"]:
            st.markdown(f"**Top relevance score:** {result['relevance_scores'][0]:.4f}")

        with st.expander(f"Retrieved & reranked {len(result['documents'])} chunks"):
            for i, (doc, score) in enumerate(zip(result["documents"], result["relevance_scores"])):
                page = doc.metadata.get("page_label", "?")
                st.markdown(f"**Chunk {i+1}** (page {page}, relevance: {score:.4f}):")
                st.text(doc.page_content[:500])