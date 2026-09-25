import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sentence_transformers import CrossEncoder
import config
from src.retrieval.hybrid_search import get_hybrid_retriever


class Reranker:
    def __init__(self):
        print(f"Loading reranker model: {config.RERANKER_MODEL}")
        self.model = CrossEncoder(config.RERANKER_MODEL)

    def rerank(self, query: str, documents: list, top_n: int = None):
        """
        Score each (query, document) pair with the cross-encoder and
        return documents sorted by relevance, trimmed to top_n.
        """
        top_n = top_n or config.TOP_K_RERANKED

        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_n]


if __name__ == "__main__":
    retriever = get_hybrid_retriever()
    reranker = Reranker()

    test_query = "what is multi-head attention?"

    # Get a WIDER net from hybrid search before reranking narrows it down
    candidates = retriever.search(test_query, top_n=10)

    print(f"\n--- Reranked results for: '{test_query}' ---")
    reranked = reranker.rerank(test_query, candidates, top_n=5)

    for i, (doc, score) in enumerate(reranked):
        print(f"\nResult {i+1} (score: {score:.4f}, page {doc.metadata.get('page_label')}):")
        print(doc.page_content[:200])