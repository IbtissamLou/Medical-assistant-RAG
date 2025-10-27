"""
Retriever that queries ChromaDB with a metadata filter (user_pdf vs pubmed),
and returns top-k chunks with their metadata for prompting.
"""

from typing import List, Dict, Optional
from core.retrieval.chroma_client import ChromaDBClient


class Retriever:
    def __init__(self, vector_store: ChromaDBClient):
        self.vs = vector_store

    def fetch(
        self,
        query: str,
        n_results: int = 5,
        source_mode: Optional[str] = None,  # "user_pdf" or "pubmed"
    ) -> List[Dict]:
        where = {"source_mode": source_mode} if source_mode else None
        results = self.vs.query(query=query, n_results=n_results, where=where)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        out = []
        for d, m in zip(docs, metas):
            out.append({"text": d, "metadata": m})
        return out
