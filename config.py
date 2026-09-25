import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Add it to your .env file.")

# --- Models ---
LLM_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Retrieval ---
TOP_K_DENSE = 10
TOP_K_SPARSE = 10
TOP_K_RERANKED = 5
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Agent ---
MAX_RETRIES = 2  # how many times the agent can rewrite+retry a query
RELEVANCE_THRESHOLD = 0.3  # grading cutoff for "is this context good enough"