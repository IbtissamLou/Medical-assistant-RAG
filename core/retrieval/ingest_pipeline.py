import os
import uuid
from typing import Optional

from .document_loader import DocumentLoader
from .text_preprocessor import TextPreprocessor
from .chunker import Chunker
from .chroma_client import ChromaDBClient


class IngestPipeline:
    def __init__(self, collection_name="medical_chunks"):
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.chunker = Chunker()
        self.vector_store = ChromaDBClient(collection_name)

    def _process_text(self, text: str, source: str, meta_extra: Optional[dict] = None):
        """Common pipeline: clean → validate → chunk → store"""
        
        cleaned = self.preprocessor.clean_text(text)
        self.preprocessor.validate(cleaned)

        print("✂️ Chunking text...")
        chunks = self.chunker.create_chunks(cleaned)

        print(f"📌 Adding {len(chunks)} chunks to ChromaDB...")

        for i, chunk in enumerate(chunks):
            unique_id = str(uuid.uuid4())
            metadata = {
                "source": source,
                "chunk_index": i,
            }
            if meta_extra:
                metadata.update(meta_extra)

            self.vector_store.add_chunk(
                chunk_id=unique_id,
                chunk_text=chunk.page_content,
                metadata=metadata,
            )

        print("✅ Ingestion completed successfully!")

    # ---------------------------------------------------------
    # 📥 1️⃣ Ingest Local File (PDF or TEXT)
    # ---------------------------------------------------------
    def ingest_from_file(self, file_path: str, source_url: str):
        print(f"📄 Loading file: {file_path}")

        if file_path.lower().endswith(".pdf"):
            raw = self.loader.load_pdf(file_path)
        else:
            raw = self.loader.load_text(file_path)

        self._process_text(
            text=raw,
            source=source_url,
            meta_extra={"file_name": os.path.basename(file_path)}
        )

    # ---------------------------------------------------------
    # 🌍 2️⃣ Ingest WHO Guideline (via Download URL)
    # ---------------------------------------------------------
    def ingest_from_who(self, download_url: str):
        print(f"🌐 Fetching WHO guideline: {download_url}")
        raw = self.loader.fetch_who_guideline(download_url)

        self._process_text(
            text=raw,
            source="WHO",
            meta_extra={"url": download_url}
        )

    # ---------------------------------------------------------
    # 🔬 3️⃣ Ingest PubMed Article By PMID
    # ---------------------------------------------------------
    def ingest_from_pubmed(self, pmid: str, api_key: Optional[str] = None):
        print(f"🔎 Fetching PubMed data for PMID: {pmid}")

        raw = self.loader.fetch_pubmed_abstract(
            pmid=pmid,
            api_key=api_key or os.getenv("PUBMED_API_KEY")
        )

        self._process_text(
            text=raw,
            source="PubMed",
            meta_extra={"pmid": pmid}
        )
