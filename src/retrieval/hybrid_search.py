import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from rank_bm25 import BM25Okapi
import config
from src.ingestion.embed_and_store import build_vectorstore
from src.ingestion.load_and_chunk import load_and_chunk_all


class HybridRetriever:
    """
    Combines dense (embedding) search and sparse (BM25 keyword) search,
    merged with Reciprocal Rank Fusion (RRF).

    Why: dense search is great at semantic similarity but weak on exact
    terms/numbers/names. BM25 is the opposite. RRF gets the best of both
    without needing to tune a blend weight.
    """

    def __init__(self, vectorstore, chunks):
        self.vectorstore = vectorstore
        self.chunks = chunks

        # Build BM25 index over tokenized chunk text
        tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, query: str, k: int):
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def sparse_search(self, query: str, k: int):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # Get top-k chunk indices by BM25 score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_indices]

    def reciprocal_rank_fusion(self, dense_results, sparse_results, k_constant: int = 60):
        """
        RRF formula: score(doc) = sum over each ranked list of 1 / (k + rank)
        Docs that show up highly ranked in BOTH lists win. This is a standard,
        parameter-light way to merge ranked lists without tuning weights.
        """
        scores = {}
        doc_lookup = {}

        for rank, doc in enumerate(dense_results):
            key = doc.page_content  # use content as dedup key
            scores[key] = scores.get(key, 0) + 1 / (k_constant + rank + 1)
            doc_lookup[key] = doc

        for rank, doc in enumerate(sparse_results):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (k_constant + rank + 1)
            doc_lookup[key] = doc

        # Sort by fused score, descending
        ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_lookup[key] for key in ranked_keys]

    def search(self, query: str, k_dense: int = None, k_sparse: int = None, top_n: int = None):
        """Run hybrid search and return fused, ranked results."""
        k_dense = k_dense or config.TOP_K_DENSE
        k_sparse = k_sparse or config.TOP_K_SPARSE
        top_n = top_n or config.TOP_K_RERANKED

        dense_results = self.dense_search(query, k_dense)
        sparse_results = self.sparse_search(query, k_sparse)
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)

        return fused_results[:top_n]


def get_hybrid_retriever():
    """Convenience function: builds/loads everything needed for hybrid search."""
    vectorstore = build_vectorstore(force_rebuild=False)
    chunks = load_and_chunk_all()  # needed for BM25 index
    return HybridRetriever(vectorstore, chunks)


if __name__ == "__main__":
    retriever = get_hybrid_retriever()

    test_query = "what is multi-head attention?"
    print(f"\n--- Hybrid search: '{test_query}' ---")
    results = retriever.search(test_query)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1} (page {doc.metadata.get('page_label')}):")
        print(doc.page_content[:200])