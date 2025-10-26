import pytest
from unittest.mock import patch
from core.retrieval.ingest_pipeline import IngestPipeline

FAKE_PUBMED_XML = """
<PubmedArticle>
  <MedlineCitation>
    <Article>
      <ArticleTitle>Test Article</ArticleTitle>
      <Abstract><AbstractText>Test content about medicine and immune response.</AbstractText></Abstract>
    </Article>
  </MedlineCitation>
</PubmedArticle>
"""

@patch("core.retrieval.document_loader.DocumentLoader.fetch_pubmed_abstract")
def test_pubmed_ingestion_with_embeddings(mock_fetch):
    mock_fetch.return_value = FAKE_PUBMED_XML

    pipeline = IngestPipeline(collection_name="test_pubmed_chunks_embed")
    pipeline.ingest_from_pubmed("99999999")


    # Check stored count
    count = pipeline.vector_store.collection.count()
    assert count > 0, "❌ No chunks stored after PubMed ingestion"

    results = pipeline.vector_store.collection.get(include=["embeddings", "metadatas", "documents"])
    
    # Validate embeddings
    assert len(results["embeddings"]) == count, "❌ Embedding count mismatch"
    assert all(len(emb) > 0 for emb in results["embeddings"]), "❌ Some embeddings are empty"

    # Verify PMID metadata presence
    assert any("pmid" in meta and meta["pmid"] == "99999999"
               for meta in results["metadatas"]), "❌ PMID not found in metadatas"

    print("✅ PubMed automated ingestion + embedding test passed!")
