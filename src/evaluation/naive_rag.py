import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from src.ingestion.embed_and_store import build_vectorstore
from src.agent.graph import get_llm, invoke_with_retry, extract_text


def naive_rag_answer(query: str, vectorstore, k: int = 5):
    """
    Simple RAG: embed query -> top-k similarity search -> stuff into prompt -> generate.
    No reranking, no relevance grading, no retries, no groundedness check.
    This represents the 'naive' baseline your agentic pipeline is compared against.
    """
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question using the context below.

Context:
{context}

Question: {query}

Answer:"""

    llm = get_llm()
    response = invoke_with_retry(llm, prompt)
    answer = extract_text(response).strip()

    return {
        "answer": answer,
        "retrieved_contexts": [doc.page_content for doc in docs],
    }


if __name__ == "__main__":
    vectorstore = build_vectorstore(force_rebuild=False)

    test_query = "What is the vanishing gradient problem and how is it addressed?"
    result = naive_rag_answer(test_query, vectorstore)

    print(f"\nQuery: {test_query}")
    print(f"\nAnswer:\n{result['answer']}")