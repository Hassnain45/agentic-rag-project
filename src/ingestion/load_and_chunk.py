import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def load_pdf(pdf_path: Path):
    """Load a single PDF and return LangChain Document objects (one per page)."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {pdf_path.name}")
    return pages


def chunk_documents(pages):
    """Split pages into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def load_and_chunk_all():
    """Load every PDF in data/raw and return a combined list of chunks."""
    pdf_files = list(config.RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {config.RAW_DATA_DIR}. Drop a PDF there first."
        )

    all_chunks = []
    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)

    print(f"Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all()
    print("\nSample chunk:")
    print(chunks[0].page_content[:300])
    print("\nMetadata:", chunks[0].metadata)