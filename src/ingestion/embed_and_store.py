import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config
from src.ingestion.load_and_chunk import load_and_chunk_all


def get_embedding_model():
    """Load the BGE embedding model. Runs locally on CPU, no API key needed."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # important for cosine similarity
    )


def build_vectorstore(force_rebuild: bool = False):
    """
    Build (or load) the ChromaDB vector store.
    If it already exists and force_rebuild is False, just load it (fast).
    Otherwise, chunk + embed + store from scratch.
    """
    embedding_model = get_embedding_model()

    if config.CHROMA_DIR.exists() and not force_rebuild:
        print(f"Loading existing vector store from {config.CHROMA_DIR}")
        vectorstore = Chroma(
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=embedding_model,
        )
        print(f"Loaded vector store with {vectorstore._collection.count()} vectors")
        return vectorstore

    if config.CHROMA_DIR.exists() and force_rebuild:
        import shutil
        print(f"Removing existing vector store at {config.CHROMA_DIR} before rebuild...")
        shutil.rmtree(config.CHROMA_DIR)

    print("Building vector store from scratch...")
    chunks = load_and_chunk_all()

    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL} (this may take a minute)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"Vector store built and saved to {config.CHROMA_DIR}")
    print(f"Total vectors stored: {vectorstore._collection.count()}")
    return vectorstore


if __name__ == "__main__":
    vectorstore = build_vectorstore(force_rebuild=True)

    # Quick sanity check: run a test query
    print("\n--- Test query ---")
    results = vectorstore.similarity_search("what is self-attention?", k=3)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1} (page {doc.metadata.get('page_label')}):")
        print(doc.page_content[:200])