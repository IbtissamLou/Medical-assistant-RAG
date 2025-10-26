"""
Purpose:
--------
Chunk long medical texts into semantically manageable segments to enable:
  - High recall during semantic search
  - Reduced hallucination in RAG LLM responses
  - Efficient embeddings storage in ChromaDB

Design Notes:
-------------
- Chunk sizes are tuned for medical content (scientific abstracts, guidelines).
- Chunk overlap preserves coherence across boundaries.
- Cleaner whitespace improves embedding consistency.
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document


class Chunker:
    def __init__(
        self,
        default_chunk_size: int = 600,
        chunk_overlap: int = 80,
    ):
        """
        Args:
            default_chunk_size: Maximum text size per chunk.
            chunk_overlap: Characters overlapped between consecutive chunks to retain context.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "? ", "! "],
            chunk_size=default_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def clean(self, text: str) -> str:
        """Basic cleaning to normalize whitespace & remove noise before chunking."""
        text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces & newlines
        return text.strip()

    def create_chunks(self, raw_text: str) -> List[Document]:
        """
        Split text into quality-controlled chunks.

        Returns:
            A list of LangChain Document objects with:
              - doc.page_content: chunk text
              - doc.metadata: auto-copied metadata if provided upstream
        """
        cleaned = self.clean(raw_text)

        chunks = self.splitter.create_documents([cleaned])

        # Filter out empty or too-short chunks
        chunks = [c for c in chunks if len(c.page_content) > 50]

        return chunks
