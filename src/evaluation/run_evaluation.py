import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from src.ingestion.embed_and_store import build_vectorstore
from src.evaluation.naive_rag import naive_rag_answer
from src.agent.graph import run_agent, get_llm

# Full test set now that the harness is confirmed working end-to-end.
SAMPLE_SIZE = 15


def load_testset():
    path = config.BASE_DIR / "data" / "processed" / "testset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_results(testset, vectorstore):
    """Run both pipelines over the test set and collect everything RAGAS needs."""
    naive_rows = []
    agentic_rows = []

    for i, item in enumerate(testset):
        question = item["question"]
        reference = item["reference_answer"]
        print(f"\n--- [{i+1}/{len(testset)}] {question}")

        print("  Running naive RAG...")
        naive_result = naive_rag_answer(question, vectorstore)
        naive_rows.append({
            "user_input": question,
            "response": naive_result["answer"],
            "retrieved_contexts": naive_result["retrieved_contexts"],
            "reference": reference,
        })

        print("  Running agentic RAG...")
        agentic_result = run_agent(question)
        agentic_rows.append({
            "user_input": question,
            "response": agentic_result["generation"],
            "retrieved_contexts": [doc.page_content for doc in agentic_result["documents"]],
            "reference": reference,
        })

    return naive_rows, agentic_rows


if __name__ == "__main__":
    testset = load_testset()[:SAMPLE_SIZE]
    print(f"Evaluating on {len(testset)} questions")

    vectorstore = build_vectorstore(force_rebuild=False)
    naive_rows, agentic_rows = collect_results(testset, vectorstore)

    # Save raw results so we don't have to re-run the (expensive) LLM calls
    # if RAGAS scoring itself needs debugging afterward.
    out_dir = config.BASE_DIR / "data" / "processed"
    with open(out_dir / "naive_results.json", "w", encoding="utf-8") as f:
        json.dump(naive_rows, f, indent=2, ensure_ascii=False)
    with open(out_dir / "agentic_results.json", "w", encoding="utf-8") as f:
        json.dump(agentic_rows, f, indent=2, ensure_ascii=False)

    print(f"\nSaved raw results to {out_dir}")
    print("Naive and agentic answers collected. Next: score with RAGAS.")