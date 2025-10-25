import pytest
from unittest.mock import patch
from core.retrieval.ingest_pipeline import IngestPipeline


FAKE_WHO_CONTENT = "WHO guideline text about infection control."


@patch("core.retrieval.document_loader.DocumentLoader.fetch_who_guideline")
def test_who_ingestion(mock_fetch):
    mock_fetch.return_value = FAKE_WHO_CONTENT

    pipeline = IngestPipeline(collection_name="test_who_chunks")
    pipeline.ingest_from_who("https://fake-who.org/guideline.pdf")

    assert pipeline.vector_store.collection.count() > 0
