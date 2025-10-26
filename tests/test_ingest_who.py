import pytest
from unittest.mock import patch
from core.retrieval.ingest_pipeline import IngestPipeline

FAKE_WHO_CONTENT = "WHO guideline text about infection control and safety procedures for hospitals."

@patch("core.retrieval.document_loader.DocumentLoader.fetch_who_guideline")
def test_who_ingestion_with_embeddings(mock_fetch):
    mock_fetch.return_value = FAKE_WHO_CONTENT

    pipeline = IngestPipeline(collection_name="test_who_chunks_embed")
    pipeline.ingest_from_who("https://fake-who.org/guideline.pdf")

    # Check documents stored
    count = pipeline.vector_store.collection.count()
    assert count > 0, "❌ No chunks stored from WHO ingestion"

    results = pipeline.vector_store.collection.get(include=["embeddings", "metadatas", "documents"])
    
    # Verify embeddings
    assert "embeddings" in results, "❌ Embeddings missing"
    assert len(results["embeddings"]) == count, "❌ Embedding count mismatch"

    assert all(len(emb) > 0 for emb in results["embeddings"]), "❌ Some embeddings are empty"
    
    # Verify metadata contains WHO
    assert any(meta["source"] == "WHO" for meta in results["metadatas"]), "❌ Metadata missing WHO source"

    print("✅ WHO automated ingestion + embedding test passed!")
