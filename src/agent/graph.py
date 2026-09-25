import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from google.genai.errors import ServerError, ClientError
import config
from src.retrieval.hybrid_search import get_hybrid_retriever
from src.retrieval.reranker import Reranker


class AgentState(TypedDict):
    original_query: str
    current_query: str
    documents: List[Document]
    relevance_scores: List[float]
    generation: str
    is_relevant: bool
    is_grounded: bool
    retrieval_attempts: int
    generation_attempts: int


# --- Shared resources, loaded once ---
_retriever = None
_reranker = None
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0,
        )
    return _llm


def get_resources():
    global _retriever, _reranker
    if _retriever is None:
        _retriever = get_hybrid_retriever()
    if _reranker is None:
        _reranker = Reranker()
    return _retriever, _reranker


def invoke_with_retry(llm, prompt, max_retries: int = 3, base_delay: float = 5.0):
    """
    Gemini's free tier can return 503 (overloaded) or 429 (rate limit).
    Retry with backoff instead of crashing the whole agent run.
    """
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except (ServerError, ClientError) as e:
            is_rate_limit = isinstance(e, ClientError) and "429" in str(e)
            is_server_error = isinstance(e, ServerError)

            if not (is_rate_limit or is_server_error):
                raise  # some other client error (e.g. bad request) — don't retry, surface it

            if attempt == max_retries - 1:
                raise

            delay = base_delay * (2 ** attempt)
            print(f"[RETRY] Gemini issue ({'rate limit' if is_rate_limit else 'overloaded'}), attempt {attempt + 1}/{max_retries}, waiting {delay}s...")
            time.sleep(delay)


def extract_text(response) -> str:
    """
    Gemini 3-series models can return content as a string OR as a list of
    parts (e.g. [{'type': 'thinking', ...}, {'type': 'text', 'text': '...'}]).
    This normalizes either shape into a plain string.
    """
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "".join(text_parts)

    return str(content)


# --- Nodes ---

def retrieve_node(state: AgentState) -> AgentState:
    print(f"\n[RETRIEVE] Query: '{state['current_query']}'")
    retriever, reranker = get_resources()

    candidates = retriever.search(state["current_query"], top_n=10)
    reranked = reranker.rerank(state["current_query"], candidates, top_n=5)

    documents = [doc for doc, score in reranked]
    scores = [float(score) for doc, score in reranked]

    print(f"[RETRIEVE] Top score: {scores[0]:.4f}" if scores else "[RETRIEVE] No results")

    state["documents"] = documents
    state["relevance_scores"] = scores
    state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
    return state


def grade_relevance_node(state: AgentState) -> AgentState:
    scores = state["relevance_scores"]

    if not scores:
        state["is_relevant"] = False
        return state

    top_score = scores[0]
    is_relevant = top_score >= config.RELEVANCE_THRESHOLD

    print(f"[GRADE] Top relevance score: {top_score:.4f} -> {'PASS' if is_relevant else 'FAIL'}")
    state["is_relevant"] = is_relevant
    return state


def rewrite_query_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = f"""The following search query did not retrieve relevant results from a document about deep learning / transformers.
Rewrite it to be clearer and more specific, using terminology likely to appear in a technical paper.

Original query: {state['current_query']}

Return ONLY the rewritten query, nothing else."""

    response = invoke_with_retry(llm, prompt)
    new_query = extract_text(response).strip()
    print(f"[REWRITE] '{state['current_query']}' -> '{new_query}'")

    state["current_query"] = new_query
    return state


def generate_node(state: AgentState) -> AgentState:
    llm = get_llm()
    context = "\n\n---\n\n".join([doc.page_content for doc in state["documents"]])

    prompt = f"""Answer the question using ONLY the context below.

Context:
{context}

Question: {state['original_query']}

Instructions:
- If the context answers the question, give a direct, confident answer.
- If the context does NOT contain enough information, say so in one clear sentence — don't hedge with phrases like "does not fully explain" if the information genuinely isn't there.
- Do not add meta-commentary about what the context does or doesn't discuss unless the question is truly unanswerable.

Answer:"""

    response = invoke_with_retry(llm, prompt)
    answer_text = extract_text(response).strip()
    print(f"[GENERATE] Answer drafted ({len(answer_text)} chars)")

    state["generation"] = answer_text
    state["generation_attempts"] = state.get("generation_attempts", 0) + 1
    return state


def grade_groundedness_node(state: AgentState) -> AgentState:
    llm = get_llm()
    context = "\n\n---\n\n".join([doc.page_content for doc in state["documents"]])

    prompt = f"""You are a strict fact-checker. Does the ANSWER below rely only on information present in the CONTEXT, with no invented facts?

CONTEXT:
{context}

ANSWER:
{state['generation']}

Respond with ONLY one word: "GROUNDED" or "UNGROUNDED"."""

    response = invoke_with_retry(llm, prompt)
    verdict = extract_text(response).strip().upper()
    is_grounded = "GROUNDED" in verdict and "UNGROUNDED" not in verdict

    print(f"[GROUND CHECK] {verdict} -> {'PASS' if is_grounded else 'FAIL'}")
    state["is_grounded"] = is_grounded
    return state


# --- Routing ---

def route_after_grading(state: AgentState) -> str:
    if state["is_relevant"]:
        return "generate"
    if state["retrieval_attempts"] >= config.MAX_RETRIES:
        print("[ROUTE] Max retrieval attempts hit — generating anyway with best-effort context")
        return "generate"
    return "rewrite_query"


def route_after_groundedness(state: AgentState) -> str:
    if state["is_grounded"]:
        return "end"
    if state["generation_attempts"] >= config.MAX_RETRIES:
        print("[ROUTE] Max generation attempts hit — returning best-effort answer with a caveat")
        return "end"
    print("[ROUTE] Answer not grounded — regenerating")
    return "generate"


# --- Graph construction ---

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_relevance", grade_relevance_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade_groundedness", grade_groundedness_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")

    graph.add_conditional_edges(
        "grade_relevance",
        route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")

    graph.add_edge("generate", "grade_groundedness")

    graph.add_conditional_edges(
        "grade_groundedness",
        route_after_groundedness,
        {"end": END, "generate": "generate"},
    )

    return graph.compile()


def run_agent(query: str):
    app = build_agent_graph()
    initial_state = {
        "original_query": query,
        "current_query": query,
        "documents": [],
        "relevance_scores": [],
        "generation": "",
        "is_relevant": False,
        "is_grounded": False,
        "retrieval_attempts": 0,
        "generation_attempts": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    test_queries = [
        "What is the vanishing gradient problem and how is it addressed?",
        "What is the capital of France?",
    ]

    for test_query in test_queries:
        print(f"\n{'='*60}\nQUERY: {test_query}\n{'='*60}")
        result = run_agent(test_query)
        print(f"\n{'='*60}\nFINAL ANSWER:\n{'='*60}")
        print(result["generation"])
        print(f"\nGrounded: {result['is_grounded']}")
        print(f"Retrieval attempts: {result['retrieval_attempts']}")
        print(f"Generation attempts: {result['generation_attempts']}")