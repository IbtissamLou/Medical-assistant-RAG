"""
LLM-based re-ranking for retrieved chunks.

Goal:
Score each chunk (0-1) based on how relevant it is to the user's question.
Irrelevant chunks get filtered out before final RAG prompt.
"""


from typing import List, Dict
from langchain.llms import Ollama


class ReRanker:
    def __init__(self, model: str = "phi3:latest"):
        self.llm = Ollama(model=model, temperature=0)

    def batch_score(self, question: str, chunks: List[Dict]) -> List[float]:
        """
        Score relevance for multiple chunks in a single LLM call.
        Output: list of numbers aligned with chunks order
        """
        numbered_chunks = "\n".join(
            f"Chunk {i}: {c['text']}" for i, c in enumerate(chunks)
        )
        prompt = f"""
You will rate each chunk for how well it helps answer the question.

Question:
{question}

Chunks:
{numbered_chunks}

Respond ONLY with a comma-separated list of numbers 0.0 to 1.0
Example: 0.2, 0.9, 0.5
"""
        try:
            raw_scores = self.llm(prompt).strip()
            return [float(v) for v in raw_scores.split(",")]
        except:
            return [0.0] * len(chunks)

    def rerank(self, question: str, retrieved: List[Dict], top_n: int = 5) -> List[Dict]:
        scores = self.batch_score(question, retrieved)
        scored = list(zip(scores, retrieved))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored[:top_n] if score >= 0.05]
