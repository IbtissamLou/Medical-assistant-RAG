import pytest
from unittest.mock import patch
from core.retrieval.ingest_pipeline import IngestPipeline


FAKE_PMIDS = ["12345", "67890"]

FAKE_PUBMED_XML = """
<PubmedArticle>
  <MedlineCitation>
    <Article>
      <ArticleTitle>Test Immunology Article</ArticleTitle>
      <Abstract><AbstractText>Immune response to pathogens.</AbstractText></Abstract>
      <Journal><Title>Test Medical Journal</Title></Journal>
    </Article>
  </MedlineCitation>
  <PubDate><Year>2024</Year></PubDate>
</PubmedArticle>
"""


@patch("core.retrieval.document_loader.DocumentLoader.search_pubmed_pmids")
@patch("core.retrieval.document_loader.DocumentLoader.fetch_pubmed_article")
def test_pubmed_topic_ingestion(mock_fetch_article, mock_search_pmids):
    # Mock API responses
    mock_search_pmids.return_value = FAKE_PMIDS
    mock_fetch_article.return_value = {
        "text": ("Test Immunology Article. The immune response is a complex biological "
                "defense mechanism involving innate and adaptive systems to protect the "
                "body from infections, pathogens, and other harmful agents. "
                "This includes antibody production, T-cell activation, and inflammatory processes."),
        "metadata": {
            "pmid": "12345",
            "title": "Test Immunology Article",
            "journal": "Test Medical Journal",
            "year": "2024",
            "source_mode": "pubmed"
        }
    }

    pipeline = IngestPipeline(collection_name="test_pubmed_chunks")

    pipeline.ingest_pubmed_by_query("immune response", retmax=2)

    count = pipeline.vector_store.count()
    assert count > 0, "❌ No chunks saved from PubMed ingestion"

    results = pipeline.vector_store.get(include=["embeddings", "metadatas", "documents"])

    # ✅ Embedding validation
    assert all(len(e) > 0 for e in results["embeddings"]), "❌ Empty embedding found"

    # ✅ Metadata validation
    assert all(meta.get("source_mode") == "pubmed" for meta in results["metadatas"])
    assert all("pmid" in meta for meta in results["metadatas"])
    assert all("journal" in meta for meta in results["metadatas"])

    print(f"✅ PubMed ingestion & embedding success — Chunks stored: {count}")
