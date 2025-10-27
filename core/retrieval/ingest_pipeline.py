"""
Purpose:
--------
Pipeline for:
  - Loading medical documents (local PDF or PubMed article)
  - Cleaning & validation
  - Chunking
  - Embedding & Vector Store persistence
"""

import os
import uuid
import datetime
from typing import Optional

from .document_loader import DocumentLoader
from .text_preprocessor import TextPreprocessor
from .chunker import Chunker
from .chroma_client import ChromaDBClient


class IngestPipeline:

    def __init__(self, collection_name="medical_chunks",use_hf_embeddings=True):
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.chunker = Chunker()
        self.vector_store = ChromaDBClient(collection_name,use_hf_embeddings=use_hf_embeddings)

    def _process(self, text: str, meta_extra: dict):
        """Clean → Validate → Chunk → Store"""
        cleaned = self.preprocessor.clean_text(text)
        self.preprocessor.validate(cleaned)

        chunks = self.chunker.create_chunks(cleaned)

        for i, chunk in enumerate(chunks):
            metadata = {
                "chunk_index": i,
                "ingested_at": datetime.datetime.utcnow().isoformat(),
            }
            metadata.update(meta_extra)

            self.vector_store.add_chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_text=chunk.page_content,
                metadata=metadata,
            )

        print(f"✅ Added {len(chunks)} chunks from: {meta_extra.get('source_mode')}")

    # ✅ Mode A — User PDF ingestion
    def ingest_pdf_file(self, file_path: str):
        print(f"📄 Ingesting PDF: {file_path}")

        raw = self.loader.load_pdf(file_path)

        self._process(
            raw,
            meta_extra={
                "source_mode": "user_pdf",
                "file_name": os.path.basename(file_path),
            },
        )

    # ✅ Mode B — PubMed ingestion by topic
    def ingest_pubmed_by_query(self, query: str, retmax: int = 5):
        print(f"🔍 Searching PubMed for: '{query}'")

        pmids = self.loader.search_pubmed_pmids(query, retmax)

        for pmid in pmids:
            # ✅ Duplication check before ingestion
            existing = self.vector_store.get(where={"pmid": pmid})
            if existing.get("ids"):
                print(f"⚠️ Skipping duplicate PMID: {pmid}")
                continue

            article = self.loader.fetch_pubmed_article(pmid)

            self._process(
                article["text"],
                meta_extra={**article["metadata"], "source_mode": "pubmed"},
            )

        print(f"✅ PubMed ingestion completed for topic: {query}")
