import streamlit as st
import os
from dotenv import load_dotenv

from core.retrieval.ingest_pipeline import IngestPipeline
from core.rag.retriever import Retriever
from core.rag.answerer import RAGAnswerer
from core.rag.llm_provider import LLMProvider  # ✅ using Ollama / LangChain provider

load_dotenv()

pipeline = IngestPipeline(
    collection_name="RAG_Assistant_pubmedbert",  # ✅ separate DB for real embeddings
    use_hf_embeddings=True                       # ✅ medical embedding model
)

retriever = Retriever(pipeline.vector_store)
rag = RAGAnswerer(retriever=retriever, llm=LLMProvider(model="llama3"))

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺")
st.title("🩺 Medical RAG Assistant")
st.write("Search evidence-based medical knowledge using your documents or PubMed.")

# ---------------------------------------
# ✅ Select Mode
# ---------------------------------------
mode = st.sidebar.radio(
    "Choose search mode",
    ["📄 My Medical Document", "🌍 General Medical Search (PubMed)"]
)

query = st.text_area(
    "Ask a medical question",
    placeholder="e.g., What are symptoms of diabetes?"
)
n_results = st.sidebar.slider("Number of results", 1, 10, 3)

st.markdown("---")

# ----------------------------------------------------------------------
# ✅ MODE A — User PDF ingestion & private RAG search
# ----------------------------------------------------------------------
if mode == "📄 My Medical Document":
    st.subheader("📄 Upload your medical PDF document")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        pdf_path = os.path.join(upload_dir, uploaded.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("✅ Processing + embedding PDF..."):
            pipeline.ingest_pdf_file(pdf_path)

        st.success("Uploaded and stored successfully ✅")

    col1, col2 = st.columns(2)

    # 🔹 Semantic Search only
    with col1:
        if st.button("🔎 Search My Document"):
            if not query:
                st.error("Please enter a question")
            else:
                with st.spinner("Searching document..."):
                    res = pipeline.vector_store.query(
                        query=query,
                        n_results=n_results,
                        where={"source_mode": "user_pdf"}
                    )
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]

                st.write("### 🧩 Retrieved Chunks")
                if not docs:
                    st.warning("No relevant information found.")
                else:
                    for d, m in zip(docs, metas):
                        st.markdown("---")
                        st.write(d)
                        st.caption(f"📁 {m.get('file_name')} | ⏱ {m.get('ingested_at')}")

    # 🔹 RAG Answer Generation
    with col2:
        if st.button("🧠 Answer with LLM (RAG) — My Document"):
            if not query:
                st.error("Please enter a question")
            else:
                with st.spinner("Generating medical answer..."):
                    out = rag.answer(query, n_results=n_results, source_mode="user_pdf")
                st.markdown("## ✅ Medical Answer")
                st.write(out["answer"])
                st.markdown("## 📚 Sources")
                for i, src in enumerate(out["sources"], start=1):
                    st.write(f"[{i}] {src}")

# ----------------------------------------------------------------------
# ✅ MODE B — PubMed ingestion, search & RAG answer
# ----------------------------------------------------------------------
elif mode == "🌍 General Medical Search (PubMed)":
    st.subheader("🌍 Search PubMed open-access medical research")

    if st.button("🔎 Fetch PubMed Articles"):
        if not query:
            st.error("Enter a topic such as 'Asthma treatment'")
        else:
            with st.spinner("Fetching + embedding PubMed articles..."):
                pipeline.ingest_pubmed_by_query(query, retmax=20)
            st.success(f"✅ PubMed articles added for: {query}")

    col1, col2 = st.columns(2)

    # 🔹 Search existing PubMed DB
    with col1:
        if st.button("📚 Search Retrieved PubMed Knowledge"):
            if not query:
                st.error("Please enter a question")
            else:
                with st.spinner("Searching PubMed knowledge..."):
                    res = pipeline.vector_store.query(
                        query=query,
                        n_results=n_results,
                        where={"source_mode": "pubmed"}
                    )
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]

                st.write("### 🔬 Retrieved PubMed Chunks")
                if not docs:
                    st.warning("No data yet — fetch PubMed first ↑")
                else:
                    for d, m in zip(docs, metas):
                        st.markdown("---")
                        st.write(d)
                        st.caption(
                            f"📚 {m.get('journal', 'Journal')} ({m.get('year', 'N/A')}) — PMID: {m.get('pmid')}"
                        )

    # 🔹 RAG Answer
    with col2:
        if st.button("🧠 Answer with LLM (RAG) — PubMed"):
            if not query:
                st.error("Please enter a question")
            else:
                with st.spinner("Synthesizing answer from PubMed..."):
                    out = rag.answer(query, n_results=n_results, source_mode="pubmed")

                st.markdown("## ✅ Medical Answer")
                st.write(out["answer"])
                st.markdown("## 📚 Sources")
                for i, src in enumerate(out["sources"], start=1):
                    st.write(f"[{i}] {src}")
