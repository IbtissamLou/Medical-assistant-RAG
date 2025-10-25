import streamlit as st
import os
from core.retrieval.ingest_pipeline import IngestPipeline

pipeline = IngestPipeline(collection_name="RAG_Assistant")

st.title("Medical RAG Assistant")
st.write("Upload medical PDF, ingest it, then query stored knowledge ✅")

# Sidebar config
query = st.text_area("Ask a medical question")
n = st.sidebar.number_input("Results", 1, 10, 3)

# File uploader (PDF)
uploaded = st.file_uploader("Upload a medical PDF", type=["pdf"])

if uploaded:
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.read())

    st.success(f"📄 File uploaded: {uploaded.name}")
    
    with st.spinner("Processing & embedding document..."):
        pipeline.ingest_from_file(path, "user_upload")
    st.success("✅ PDF ingested and stored successfully!")

# Query ChromaDB
if st.button("Search DB"):
    res = pipeline.vector_store.query(query, n_results=n)
    st.write(f"🔍 Found {len(res['documents'][0])} matches:")
    
    for text, meta in zip(res["documents"][0], res["metadatas"][0]):
        st.markdown("---")
        st.markdown(f"📌 **Source:** {meta['source']}")
        #st.markdown(f"📌 **Chunk:** {meta['chunk_index']}")
        if "file_name" in meta:
            st.markdown(f"📄 File: {meta['file_name']}")
        st.write(text)
