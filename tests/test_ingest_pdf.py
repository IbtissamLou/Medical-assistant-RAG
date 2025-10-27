import os
import pytest
from core.retrieval.ingest_pipeline import IngestPipeline


@pytest.fixture(scope="module")
def sample_pdf_path() -> str:
    """Returns path to a valid test PDF file."""
    path = "data/raw/Anatomy.pdf"
    assert os.path.exists(path), f"Missing test PDF: {path}"
    return path


def test_pdf_ingestion_chunking_embedding(sample_pdf_path):
    pipeline = IngestPipeline(collection_name="test_pdf_chunks",use_hf_embeddings=False)

    # Ingest the file
    pipeline.ingest_pdf_file(sample_pdf_path)


    # Ensure chunks were stored
    count = pipeline.vector_store.count()
    assert count > 0, "❌ No chunks stored"

    results = pipeline.vector_store.get(include=["embeddings", "metadatas", "documents"])

    # ✅ Validate chunking and embedding
    assert len(results["documents"]) == count, "❌ Missing documents"
    assert len(results["embeddings"]) == count, "❌ Missing embeddings"
    assert all(len(e) > 0 for e in results["embeddings"]), "❌ Empty embedding found"

    # ✅ Metadata checks
    assert all(meta.get("source_mode") == "user_pdf" for meta in results["metadatas"])
    assert all("file_name" in meta for meta in results["metadatas"])

    print(f"✅ PDF Test Passed: {count} chunks stored with embeddings & metadata ✅")
