import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.agent.graph import get_llm
from src.ingestion.embed_and_store import get_embedding_model


def load_results(filename: str):
    path = config.BASE_DIR / "data" / "processed" / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_pipeline(rows, evaluator_llm, evaluator_embeddings, label: str):
    print(f"\nScoring {label} ({len(rows)} rows)...")

    dataset = EvaluationDataset.from_list(rows)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()
    print(f"\n{label} results:")
    print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())

    return result


if __name__ == "__main__":
    naive_rows = load_results("naive_results.json")
    agentic_rows = load_results("agentic_results.json")

    evaluator_llm = LangchainLLMWrapper(get_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())

    naive_scores = score_pipeline(naive_rows, evaluator_llm, evaluator_embeddings, "NAIVE RAG")
    agentic_scores = score_pipeline(agentic_rows, evaluator_llm, evaluator_embeddings, "AGENTIC RAG")

    naive_df = naive_scores.to_pandas()
    agentic_df = agentic_scores.to_pandas()

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        naive_avg = naive_df[metric].mean()
        agentic_avg = agentic_df[metric].mean()
        diff = agentic_avg - naive_avg
        print(f"{metric:20s}  naive: {naive_avg:.3f}   agentic: {agentic_avg:.3f}   diff: {diff:+.3f}")

    # Save full per-row scores so we can inspect exactly which questions
    # dragged any metric up or down for either pipeline.
    out_dir = config.BASE_DIR / "data" / "processed"
    naive_df.to_csv(out_dir / "naive_scores.csv", index=False)
    agentic_df.to_csv(out_dir / "agentic_scores.csv", index=False)
    print(f"\nPer-row scores saved to {out_dir}")