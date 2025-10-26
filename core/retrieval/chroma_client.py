"""
chroma_client.py

Purpose:
--------
Thin wrapper around a persistent ChromaDB collection used by the Medical RAG Assistant.
It centralizes:
  - Client/collection creation with a chosen embedding function
  - Adding chunks (text + metadata)
  - Querying with optional metadata filters (e.g., search only user PDFs vs. PubMed)
  - Small convenience helpers for inspection/maintenance

Why this matters:
-----------------
Keeping vector-store access in one place makes it easier to:
  - Swap embedding models later (e.g., BAAI/bge-m3) without touching the rest of the code
  - Enforce consistent metadata (e.g., source/source_mode) across the app
  - Support the two clear modes designed:
       1) "user_pdf" — private uploads
       2) "pubmed"   — open-access PubMed articles retrieved by topic
"""

from typing import Any, Dict, Iterable, List, Optional
import chromadb
from chromadb.utils import embedding_functions


class ChromaDBClient:
    def __init__(
        self,
        collection_name: str = "medical_chunks",
        persist_path: str = "./chroma_db",
        use_hf_embeddings: bool = False,
        hf_model_name: str = "BAAI/bge-m3",
    ) -> None:
        """
        Args:
            collection_name: Name of the Chroma collection to use/create.
            persist_path: Filesystem path where Chroma persists data.
            use_hf_embeddings: If True, use HuggingFace embeddings; otherwise DefaultEmbeddingFunction.
            hf_model_name: HF model to use when use_hf_embeddings=True.

        """
        # Persistent client → survives app restarts
        self.client = chromadb.PersistentClient(path=persist_path)

        # Choose embedding function
        if use_hf_embeddings:
            self.embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
                model_name=hf_model_name
            )
        else:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Create or get the target collection.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )

    # -----------------------------
    # Write operations
    # -----------------------------
    def add_chunk(self, chunk_id: str, chunk_text: str, metadata: Dict[str, Any]) -> None:
        """
        Add a single text chunk + metadata to the collection.
        Embeddings are created automatically via the embedding_function bound to the collection.
        """
        self.collection.add(
            ids=[chunk_id],
            documents=[chunk_text],
            metadatas=[metadata],
        )

    def add_chunks(
        self,
        chunk_ids: Iterable[str],
        chunk_texts: Iterable[str],
        metadatas: Iterable[Dict[str, Any]],
    ) -> None:
        """Batch insert helper."""
        self.collection.add(
            ids=list(chunk_ids),
            documents=list(chunk_texts),
            metadatas=list(metadatas),
        )

    # -----------------------------
    # Read / query operations
    # -----------------------------
    def query(
        self,
        query: str,
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Semantic search over the collection.

        Args:
            query: Natural language search string.
            n_results: Number of neighbor chunks to retrieve.
            where: Optional metadata filter (e.g., {"source_mode": "user_pdf"} or {"source": "PubMed"}).
            include: Which fields to include in the response. Defaults to ["documents", "metadatas", "distances"].

        Returns:
            A Chroma result dict: {"ids": [...], "documents": [...], "metadatas": [...], "distances": [...]}
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,          
            include=include,
        )

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch items directly by ids or metadata filter.
        Useful for dedup checks, debugging, and admin UI.

        Example:
            get(where={"pmid": "12345"})
        """
        if include is None:
            include = ["metadatas"] # ✅ Only need metadata to check duplicates
        return self.collection.get(ids=ids, where=where, include=include)

    def count(self) -> int:
        """Number of items in the collection."""
        return self.collection.count()

    # -----------------------------
    # Maintenance operations
    # -----------------------------
    def delete_where(self, where: Dict[str, Any]) -> None:
        """
        Delete items by metadata filter.
        Example:
            delete_where({"source_mode": "user_pdf"})
        """
        self.collection.delete(where=where)
