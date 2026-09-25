# Agentic RAG over "Dive into Deep Learning"

A Retrieval-Augmented Generation system that goes beyond the standard "chunk → embed → retrieve → generate" pipeline. This implementation adds hybrid retrieval, cross-encoder reranking, and a self-correcting agentic loop that grades its own retrieval quality and fact-checks its own answers before responding — refusing to answer rather than hallucinating when the source material doesn't support a confident answer.

Built over the 1,151-page open-source textbook *Dive into Deep Learning* (D2L).

## Why this exists

Most introductory RAG projects follow a fixed pipeline: embed a query, pull the top-k similar chunks, stuff them into a prompt, generate an answer. This works for easy questions but breaks down in two common ways — retrieval returns irrelevant chunks for ambiguous or out-of-scope queries, and the LLM confidently fabricates an answer anyway.

This project treats retrieval and generation as steps an *agent* actively supervises, rather than a fixed pipeline it blindly executes.

## Architecture
Query
|
v
[1] Retrieve (Hybrid: BM25 + Dense embeddings, merged via Reciprocal Rank Fusion)
|
v
[2] Rerank (Cross-Encoder re-scores candidates)
|
v
[3] Grade Relevance
|-- FAIL --> Rewrite Query --> back to [1] (bounded retries)
|
PASS
v
[4] Generate Answer
|
v
[5] Grade Groundedness
|-- UNGROUNDED --> back to [4], regenerate (bounded retries)
|
GROUNDED
v
Final Answer



**Retrieval**: dense (embedding) search and BM25 (keyword) search run in parallel and are merged with Reciprocal Rank Fusion, so the system doesn't lose exact-term matches (numbers, names, acronyms) the way pure embedding search often does.

**Reranking**: a cross-encoder re-scores the top candidates by jointly encoding the query and each chunk together (rather than comparing separately-embedded vectors), giving a much more precise relevance signal before anything reaches the LLM.

**Corrective loop**: if the top reranked result scores below a relevance threshold, the agent rewrites the query and retries (bounded retries) instead of generating from weak context.

**Groundedness check**: after generating an answer, a second LLM call verifies every claim is actually supported by the retrieved context. If not, the agent regenerates (bounded retries) rather than returning an unverified answer.

## Tech stack

| Component | Choice |
|---|---|
| Orchestration | LangGraph (state machine / agentic loop) |
| Dense embeddings | `BAAI/bge-base-en-v1.5` (local, CPU) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| Vector store | ChromaDB (persistent, local) |
| LLM | Gemini (`gemini-3.1-flash-lite`) |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision/recall) |
| UI | Streamlit |

## Evaluation: Agentic vs. Naive RAG

A 15-question test set was auto-generated from the book, then scored with RAGAS across both a naive RAG baseline (plain top-k similarity search, no reranking/grading/retries) and this agentic pipeline.

| Metric | Naive RAG | Agentic RAG | Δ |
|---|---|---|---|
| Faithfulness | 1.000 | 1.000 | +0.000 |
| Answer Relevancy | 0.861 | 0.748 | −0.113 |
| Context Precision | 0.683 | 0.750 | **+0.067** |
| Context Recall | 0.857 | 0.950 | **+0.093** |

**What this shows**: the agentic pipeline retrieves more of the relevant material (higher recall) with less noise (higher precision) — direct evidence that hybrid search + reranking improves retrieval quality over plain similarity search. Faithfulness tied at a perfect score for both on this test set.

**The honest trade-off**: answer relevancy is lower for the agentic pipeline. This is a real, explainable side effect of the generation prompt explicitly instructing the model to decline rather than hedge-fabricate when context is insufficient — RAGAS's relevancy metric can penalize more cautious, qualified answers even when their content is factually correct. This was confirmed by inspecting individual low-scoring rows (see `data/processed/agentic_scores.csv`).

### Example: self-correction caught mid-run

On one test question, the agent's first generated answer failed its own groundedness check and was automatically regenerated:

[RETRIEVE] Top score: 0.9863
[GRADE] Top relevance score: 0.9863 -> PASS
[GENERATE] Answer drafted (317 chars)
[GROUND CHECK] UNGROUNDED -> FAIL
[ROUTE] Answer not grounded — regenerating
[GENERATE] Answer drafted (506 chars)
[GROUND CHECK] GROUNDED -> PASS


A naive pipeline would have returned the first (ungrounded) answer with no verification step.

### Example: honest refusal instead of hallucination

Query: *"What is the capital of France?"* (deliberately out of the book's domain)

[RETRIEVE] Top score: 0.0011
[GRADE] Top relevance score: 0.0011 -> FAIL
[REWRITE] 'What is the capital of France?' -> 'What is the capital of France?'
[RETRIEVE] Top score: 0.0011
[GRADE] Top relevance score: 0.0011 -> FAIL
[ROUTE] Max retrieval attempts hit — generating anyway with best-effort context
[GENERATE] Answer drafted (80 chars)
[GROUND CHECK] GROUNDED -> PASS

Final answer: "The provided context does not contain enough information to answer the question."


## Setup

**1. Clone and set up the environment**

```bash
git clone https://github.com/hassnain45/agentic-rag-project.git
cd agentic-rag-project
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
```

**2. Add your Gemini API key**

Create a `.env` file in the project root:
GOOGLE_API_KEY=your_key_here

Get a free key at aistudio.google.com/apikey.

**3. Add a source PDF**

Drop any PDF into `data/raw/`. To use the same book as this project:

```bash
curl.exe -L "https://d2l.ai/d2l-en.pdf" -o "data/raw/dive_into_deep_learning.pdf"
```

**4. Build the vector store**

```bash
python src/ingestion/embed_and_store.py
```

**5. Run the agent from the command line**

```bash
python src/agent/graph.py
```

**6. Or launch the interactive dashboard**

```bash
streamlit run src/ui/dashboard.py
```

## Project structure
agentic-rag-project/
├── data/
│ ├── raw/ # Source PDFs
│ └── processed/ # Generated test sets, evaluation results
├── src/
│ ├── ingestion/ # PDF loading, chunking, embedding
│ ├── retrieval/ # Hybrid search (BM25 + dense + RRF), reranking
│ ├── agent/ # LangGraph agentic loop
│ ├── evaluation/ # RAGAS harness, naive RAG baseline
│ └── ui/ # Streamlit dashboard
├── requirements.txt
└── config.py # Central configuration


## Known limitations

- Runs on Gemini's free tier, which imposes daily request quotas — evaluation runs and heavy interactive use can hit rate limits (handled with exponential backoff, but not eliminated).
- Reranking runs on CPU, adding latency per query compared to a pure-embedding pipeline.
- The relevance threshold (`config.py`) was empirically tuned on this dataset/reranker combination and may need adjustment for other document types.

## Possible extensions

- Multi-modal retrieval over the book's figures/diagrams
- Query decomposition for multi-part questions
- Swap Gemini for a local model to remove rate-limit dependency

