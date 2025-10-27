"""
RAG answer orchestration:
- retrieve top-k chunks
- build medical-safe prompt with numbered citations
- call LLM
- return answer text + a normalized list of sources for UI
"""

from typing import Dict, Any, List, Optional
from core.rag.retriever import Retriever
from core.rag.prompt import build_prompt
from core.rag.llm_provider import LLMProvider


class RAGAnswerer:
    def __init__(self, retriever: Retriever, llm: Optional[LLMProvider] = None):
        self.retriever = retriever
        self.llm = llm or LLMProvider()

    def answer(
        self,
        question: str,
        n_results: int = 5,
        source_mode: Optional[str] = None,  # "user_pdf" or "pubmed"
    ) -> Dict[str, Any]:
        # 1) Retrieve
        chunks = self.retriever.fetch(
            query=question, n_results=n_results, source_mode=source_mode
        )

        # If nothing retrieved, fail gracefully
        if not chunks:
            return {
                "answer": "I don't have enough evidence in the provided sources.",
                "sources": [],
                "chunks": [],
            }

        # 2) Prompt
        prompt = build_prompt(question, chunks)

        # 3) LLM call
        completion = self.llm.generate(system=prompt["system"], user=prompt["user"])

        # 4) Prepare UI-friendly sources list
        sources: List[Dict[str, str]] = []
        for c in chunks:
            m = c["metadata"]
            if m.get("source_mode") == "pubmed":
                label = f"PMID {m.get('pmid', '')} — {m.get('journal', 'Journal')} ({m.get('year', 'N/A')})"
            else:
                label = f"{m.get('file_name', 'user_pdf')}"
            sources.append(label)

        return {"answer": completion, "sources": sources, "chunks": chunks}
