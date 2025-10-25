import os
import pytest
from core.retrieval.ingest_pipeline import IngestPipeline
from reportlab.pdfgen import canvas


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    file_path = tmp_path_factory.mktemp("data") / "sample.pdf"
    c = canvas.Canvas(str(file_path))
    c.drawString(100, 750, "This is a test medical document about vaccines.")
    c.save()
    return str(file_path)


def test_ingest_local_pdf(sample_pdf):
    pipeline = IngestPipeline(collection_name="test_file_chunks")

    pipeline.ingest_from_file(
        file_path=sample_pdf,
        source_url="user_upload"
    )

    # Basic assertion: at least one chunk stored
    assert pipeline.vector_store.collection.count() > 0
