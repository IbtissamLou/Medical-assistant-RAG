import os
import pytest
from core.retrieval.ingest_pipeline import IngestPipeline


@pytest.fixture(scope="module")
def anatomy_pdf_path() -> str:
    """Return path to the Anatomy.pdf file in the project folder"""
    path = os.path.join("data", "raw", "Anatomy.pdf")
    assert os.path.exists(path), f"Test file missing: {path}"
    return path


def test_ingest_local_pdf_with_embeddings(anatomy_pdf_path):
    pipeline = IngestPipeline(collection_name="test_file_chunks_embed")

    # Run ingestion
    pipeline.ingest_from_file(
        file_path=anatomy_pdf_path,
        source_url="user_upload"
    )

    # Check document count
    stored_count = pipeline.vector_store.collection.count()
    assert stored_count > 0, "❌ No chunks stored in ChromaDB"
    print(f"✅ Stored chunks count: {stored_count}")

    # Retrieve stored embeddings and docs to validate
    results = pipeline.vector_store.collection.get(include=["embeddings", "documents"])

    # Test embedding presence and shape
    assert "embeddings" in results, "❌ Embeddings missing from stored data"
    assert len(results["embeddings"]) == stored_count, "❌ Embeddings count mismatch"
    assert all(len(emb) > 0 for emb in results["embeddings"]), "❌ Found empty embedding vector"

    print("✅ Embeddings generated and stored correctly!")


