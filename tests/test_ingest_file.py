import os
import pytest
from core.retrieval.ingest_pipeline import IngestPipeline


@pytest.fixture(scope="module")
def anatomy_pdf_path() -> str:
    """Return path to the Anatomy.pdf file in the project folder"""
    path = os.path.join("data", "raw", "Anatomy.pdf")  
    assert os.path.exists(path), f"Test file missing: {path}"
    return path


def test_ingest_local_pdf(anatomy_pdf_path):
    pipeline = IngestPipeline(collection_name="test_file_chunks")

    pipeline.ingest_from_file(
        file_path=anatomy_pdf_path,
        source_url="user_upload"
    )

    stored_count = pipeline.vector_store.collection.count()
    print(f"✅ Stored chunks count: {stored_count}")

    # Basic assertion: at least one chunk stored
    assert stored_count > 0

