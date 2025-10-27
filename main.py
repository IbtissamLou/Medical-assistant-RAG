"""
Streamlit UI for the Medical RAG Assistant
------------------------------------------
Two separate knowledge search modes:

Mode A: User PDF Upload → Private Retrieval Only
Mode B: PubMed Topic Search → Open-Access Medical Knowledge

This ensures clean separation between private and public data sources.
"""

import streamlit as st
import os
from core.retrieval.ingest_pipeline import IngestPipeline

pipeline = IngestPipeline(collection_name="RAG_Assistant_pubmedbert",use_hf_embeddings=True)

st.title("🩺 Medical RAG Assistant")
st.write("Choose a mode to get started 👇")

# --------------------------------------
# ✅ MODE SELECTOR
# --------------------------------------
mode = st.sidebar.radio(
    "Select Search Mode",
    ["📄 My Medical Document", "🌍 General Medical Search (PubMed)"]
)

query = st.text_area(
    "Ask a medical question",
    placeholder="e.g., What are the symptoms of diabetes?"
)
n_results = st.sidebar.slider("Number of results", 1, 10, 3)

st.markdown("---")

# --------------------------------------
# ✅ MODE A — USER PDF KNOWLEDGE
# --------------------------------------
if mode == "📄 My Medical Document":
    st.subheader("📄 Upload a medical PDF to enable private search")

    uploaded = st.file_uploader("Upload your document", type=["pdf"])

    if uploaded:
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, uploaded.name)

        with open(file_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("Processing + embedding your document..."):
            pipeline.ingest_pdf_file(file_path)

        st.success("✅ Document added to your knowledge base!")

    if st.button("🔍 Search My Document"):
        if not query:
            st.error("Please write a question")
        else:
            with st.spinner("Searching your uploaded document..."):
                results = pipeline.vector_store.query(
                    query=query,
                    n_results=n_results,
                    where={"source_mode": "user_pdf"}
                )

            st.write("### Results:")
            if not results["documents"][0]:
                st.warning("No relevant info in your document.")
            else:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    st.markdown("----")
                    st.write(doc)
                    st.caption(f"📌 File: {meta.get('file_name')} | ⏱️ {meta.get('ingested_at')}")

# --------------------------------------
# ✅ MODE B — PubMed PUBLIC KNOWLEDGE
# --------------------------------------
elif mode == "🌍 General Medical Search (PubMed)":
    st.subheader("Search PubMed Open-Access Articles")

    if st.button("🔎 Search PubMed & Add Articles"):
        if not query:
            st.error("Enter a topic like 'Hypertension treatment'")
        else:
            with st.spinner("Fetching relevant PubMed articles..."):
                pipeline.ingest_pubmed_by_query(query, retmax=5)
            st.success(f"✅ Added PubMed knowledge for: {query}")

    if st.button("📚 Search Knowledge Base"):
        with st.spinner("Searching PubMed knowledge..."):
            results = pipeline.vector_store.query(
                query=query,
                n_results=n_results,
                where={"source_mode": "pubmed"}
            )

        st.write(f"### Retrieved PubMed Information:")
        if not results["documents"][0]:
            st.warning("Try getting PubMed data first ⬆️")
        else:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                st.markdown("----")
                st.write(doc)
                st.caption(
                    f"📚 {meta.get('journal', 'Unknown Journal')} "
                    f"({meta.get('year', 'N/A')}) — PMID: {meta.get('pmid')}"
                )
