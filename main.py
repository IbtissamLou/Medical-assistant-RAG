import streamlit as st
import os
from dotenv import load_dotenv

from core.retrieval.ingest_pipeline import IngestPipeline
from core.rag.retriever import Retriever
from core.rag.answerer import RAGAnswerer
from core.rag.llm_provider import LLMProvider

load_dotenv()

# ✅ Initialize RAG pipeline 
pipeline = IngestPipeline(
    collection_name="Medi_RAG_Assistant",
    use_hf_embeddings=True
)
retriever = Retriever(pipeline.vector_store)
rag = RAGAnswerer(retriever=retriever, llm=LLMProvider(model="phi3:latest"))

# ✅ Page setup + branding
st.set_page_config(page_title="MediChat Assistant", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main {background-color: #F7F9FB;}
    h1 {color: #0A4DA6;}
    .stTextInput, .stTextArea {border-radius: 10px;}
    .source-box {font-size: 13px; color: #3A3A3A; background: #E9F5FF; padding: 8px; border-radius: 6px; margin-bottom: 4px;}
</style>
""", unsafe_allow_html=True)

st.title("🩺 MediChat — Your Medical Knowledge Assistant")
st.write("Find trustworthy, evidence-based medical information — instantly.")

# ✅ Search mode selector (left sidebar stays minimal)
mode = st.sidebar.selectbox(
    "Select data source:",
    ["📄 Use My Uploaded Document", "🌍 Search PubMed Medical Research"]
)

n_results = st.sidebar.slider("Max Sources to Use", 1, 10, 3)

# ✅ Main user question box
query = st.text_area(
    "Ask your medical question here 👇",
    placeholder="e.g., What are the symptoms of hypertension?"
)

st.markdown("---")
document_section = st.empty()  # to update dynamically

# ----------------------------------------------------------------------
# ✅ MODE A — PDF Upload Assistant
# ----------------------------------------------------------------------
if mode == "📄 Use My Uploaded Document":

    document_section.subheader("📄 Upload a medical document")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        os.makedirs("data/uploads", exist_ok=True)
        pdf_path = f"data/uploads/{uploaded.name}"

        with open(pdf_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("Extracting and processing document... ⏳"):
            pipeline.ingest_pdf_file(pdf_path)

        st.success("✅ Document successfully added to your knowledge base!")

    # ✅ Conversational button
    if st.button("💬 Ask MediChat"):
        if not query:
            st.error("Please enter your medical question.")
        else:
            with st.spinner("Analyzing your document and medical knowledge..."):
                out = rag.answer(query, n_results=n_results, source_mode="user_pdf")

            st.markdown("### ✅ MediChat Answer")
            st.write(out["answer"])

            if out["sources"]:
                st.markdown("### 📌 Information Sources")
                for i, src in enumerate(out["sources"], start=1):
                    st.markdown(f'<div class="source-box">[{i}] {src}</div>',
                                unsafe_allow_html=True)

# ----------------------------------------------------------------------
# ✅ MODE B — PubMed Knowledge Retrieval
# ----------------------------------------------------------------------
elif mode == "🌍 Search PubMed Medical Research":

    if st.button("📥 Fetch PubMed Evidence"):
        if not query:
            st.error("Please describe your medical topic.")
        else:
            with st.spinner("Searching PubMed & indexing medical knowledge..."):
                pipeline.ingest_pubmed_by_query(query, retmax=15)
            st.success("✅ PubMed evidence added.")

    if st.button("💬 Ask MediChat"):
        if not query:
            st.error("Please enter your medical question.")
        else:
            with st.spinner("Retrieving high-confidence evidence..."):
                out = rag.answer(query, n_results=n_results, source_mode="pubmed")

            st.markdown("### ✅ MediChat Answer")
            st.write(out["answer"])

            if out["sources"]:
                st.markdown("### 🧾 References")
                for i, src in enumerate(out["sources"], start=1):
                    st.markdown(f'<div class="source-box">[{i}] {src}</div>',
                                unsafe_allow_html=True)

# ✅ Footer - safety note
st.markdown("---")
st.caption("⚠️ This tool provides information from verified medical literature, but does not replace professional medical advice.")
