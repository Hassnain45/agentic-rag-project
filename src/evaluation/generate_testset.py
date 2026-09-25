import sys
import json
import random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ingestion.load_and_chunk import load_and_chunk_all
from src.agent.graph import get_llm, invoke_with_retry, extract_text
import config

NUM_QUESTIONS = 15
MIN_CHUNK_LENGTH = 300  # skip very short/sparse chunks (tables, headers, etc.)


def generate_qa_pair(llm, chunk_text: str):
    """Ask the LLM to write one question + reference answer grounded in this chunk."""
    prompt = f"""You are creating a test question for a Deep Learning Q&A system.

Given the passage below, write:
1. ONE clear, specific question that this passage directly answers
2. A concise reference answer (2-4 sentences) using ONLY information from the passage

Passage:
{chunk_text}

Respond in EXACTLY this format, nothing else:
QUESTION: <your question>
ANSWER: <your answer>"""

    response = invoke_with_retry(llm, prompt)
    text = extract_text(response).strip()

    try:
        question_part = text.split("QUESTION:")[1].split("ANSWER:")[0].strip()
        answer_part = text.split("ANSWER:")[1].strip()
        return question_part, answer_part
    except IndexError:
        return None, None


def build_testset():
    print("Loading chunks...")
    chunks = load_and_chunk_all()

    # Filter to reasonably substantial chunks so questions are meaningful
    good_chunks = [c for c in chunks if len(c.page_content) >= MIN_CHUNK_LENGTH]
    print(f"{len(good_chunks)} chunks qualify (out of {len(chunks)} total)")

    # Sample randomly across the book for topic diversity
    random.seed(42)  # reproducible sample
    sampled_chunks = random.sample(good_chunks, min(NUM_QUESTIONS, len(good_chunks)))

    llm = get_llm()
    testset = []

    for i, chunk in enumerate(sampled_chunks):
        print(f"Generating Q&A {i+1}/{len(sampled_chunks)}...")
        question, answer = generate_qa_pair(llm, chunk.page_content)

        if question and answer:
            testset.append({
                "question": question,
                "reference_answer": answer,
                "source_page": chunk.metadata.get("page_label"),
                "source_chunk": chunk.page_content,
            })
        else:
            print(f"  Skipped (couldn't parse LLM output)")

    output_path = config.BASE_DIR / "data" / "processed" / "testset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(testset)} Q&A pairs to {output_path}")
    return testset


if __name__ == "__main__":
    testset = build_testset()
    print("\n--- Sample ---")
    for item in testset[:3]:
        print(f"\nQ: {item['question']}")
        print(f"A: {item['reference_answer']}")
        print(f"(page {item['source_page']})")