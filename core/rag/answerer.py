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
from core.rag.reranker import ReRanker


class RAGAnswerer:
    def __init__(self, retriever: Retriever, llm: Optional[LLMProvider] = None):
        self.retriever = retriever
        self.llm = llm or LLMProvider()
        self.reranker = ReRanker(model="phi3:latest")

    def answer(
        self,
        question: str,
        n_results: int = 5,
        source_mode: Optional[str] = None,
    ) -> Dict[str, Any]:

        # ✅ Retrieve semantic neighbors
        raw_chunks = self.retriever.fetch(
            query=question,
            n_results=min(n_results, 5) if source_mode == "pubmed" else n_results,
            source_mode=source_mode
        )
        print(f"Retrieved {len(raw_chunks)} chunks before reranking")

        # ✅ If PubMed → apply reranking logic
        if source_mode == "pubmed" and raw_chunks:
            chunks = self.reranker.rerank(question, raw_chunks, top_n=n_results)
            print(f"PubMed: kept {len(chunks)} chunks after reranking")

            # ✅ Soft fallback if reranker too strict
            if len(chunks) == 0:
                print("⚠️ Reranker removed everything → fallback to raw chunks")
                chunks = raw_chunks[:n_results]
        else:
            chunks = raw_chunks  # ✅ PDF case = direct use

        # ✅ If NOTHING retrieved → graceful fail
        if not chunks:
            return {
                "answer": "I don't have enough evidence to answer your question based on the available data.",
                "sources": [],
                "chunks": [],
            }

        # ✅ Prompt with selected chunks
        prompt = build_prompt(question, chunks)

        # ✅ LLM generation with safe fallback
        try:
            completion = self.llm.generate(system=prompt["system"], user=prompt["user"])
        except Exception as e:
            print(f"❌ LLM error: {e}")
            completion = "⚠️ Unable to generate answer due to model error."

        # ✅ Map metadata to human-readable sources
        sources: List[str] = []
        for c in chunks:
            m = c["metadata"]
            if m.get("source_mode") == "pubmed":
                label = f"PMID {m.get('pmid', '')} — {m.get('journal', 'Journal')} ({m.get('year', 'N/A')})"
            else:
                label = f"{m.get('file_name', 'user_pdf')}"
            sources.append(label)

        return {"answer": completion, "sources": sources, "chunks": chunks}
