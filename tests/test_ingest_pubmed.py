import pytest
from unittest.mock import patch
from core.retrieval.ingest_pipeline import IngestPipeline


FAKE_PUBMED_XML = """
<PubmedArticle>
  <MedlineCitation>
    <Article>
      <ArticleTitle>Test Article</ArticleTitle>
      <Abstract><AbstractText>Test content about medicine.</AbstractText></Abstract>
    </Article>
  </MedlineCitation>
</PubmedArticle>
"""


@patch("core.retrieval.document_loader.DocumentLoader.fetch_pubmed_abstract")
def test_pubmed_ingestion(mock_fetch):
    mock_fetch.return_value = FAKE_PUBMED_XML

    pipeline = IngestPipeline(collection_name="test_pubmed_chunks")
    pipeline.ingest_from_pubmed("12345678")

    assert pipeline.vector_store.collection.count() > 0
