# 🩺 Medical RAG Assistant  
**Evidence-Based Clinical Knowledge Assistant for Healthcare Professionals**

---

## 📌 Project Description

This project is a **medical knowledge assistant** designed for **healthcare professionals** who require **fast, reliable, and fully traceable clinical information** at the point of care.

Clinicians often lose valuable time searching for trustworthy evidence across fragmented sources, which can delay decision-making and introduce variability in care. This system addresses that challenge by delivering **concise, evidence-backed medical answers**, grounded strictly in validated medical literature and private clinical documents.

The assistant is built using a **Retrieval-Augmented Generation (RAG)** architecture to ensure that **every response is supported by retrieved medical evidence**. The system explicitly refuses to speculate beyond available sources, significantly reducing hallucination risk and aligning with expectations in regulated healthcare environments.

---

## 🎯 Target Users

- Physicians and clinicians  
- Nurses and allied healthcare professionals  
- Clinical researchers  
- Medical students and trainees  

---

## ✅ Key Guarantees

- **No hallucinations**: the system never invents medical facts  
- **Full traceability**: every answer is linked to explicit sources  
- **Privacy-first design**: private PDFs never leave the user’s environment  
- **Evidence transparency**: citations are always visible  
- **Fast responses**: optimized for near real-time clinical usage  

---

## 🧠 System Architecture

### 1. Retrieval-Augmented Generation (RAG)

The assistant relies on RAG to tightly couple information retrieval with answer generation:

1. **Retrieve** relevant medical passages from trusted sources  
2. **Ground** the LLM strictly within retrieved evidence  
3. **Generate** structured medical answers with citations  
4. **Refuse** to answer if evidence is insufficient  

This design ensures **clinical responsibility and explainability**.

---

### 2. Supported Knowledge Sources

#### 📄 Private Medical Documents (PDF)
- Secure ingestion of user-uploaded medical PDFs  
- Suitable for proprietary research, guidelines, or internal clinical notes  
- No data is sent externally  
- High recall is prioritized to avoid losing critical context  

#### 🌍 PubMed Open-Access Literature
- Ingestion of open-access PubMed articles  
- Validated licensing and provenance  
- Rich metadata: PMID, journal, year, ingestion time  
- Optimized for evidence-based medicine use cases  

---

### 3. Preprocessing & Chunking

- Documents are cleaned and validated for medical relevance  
- Chunking is performed using `RecursiveCharacterTextSplitter`  
- Paragraph and sentence boundaries are preserved  
- Prevents loss of meaning common in naive fixed-size chunking  

This significantly improves retrieval precision, especially for short PubMed abstracts.

---

### 4. Embeddings & Vector Store

- **Embedding model**: **PubMedBERT**  
  - Domain-specific biomedical language representation  
  - Captures clinical terminology and relationships better than general-purpose models  

- **Vector database**: **ChromaDB**  
  - Stores embeddings and rich metadata  
  - Enables semantic search with full provenance tracking  

Automated unit tests validate:
- Ingestion correctness  
- Chunk integrity  
- Embedding generation  
- Metadata attachment  

This is essential for reproducibility and compliance.

---

## 🤖 Medical Question Answering Pipeline

### Retrieval
- Relevant passages are retrieved from:
  - PubMed (open-access)  
  - User PDFs (private)  

- For PubMed:
  - Medical synonym expansion  
  - Query reformulation when input is vague  
  - Dense retrieval for high recall  
  - LLM-based re-ranking for precision  

- Re-ranking is **applied only to PubMed**, not to private PDFs, to preserve recall.

---

### Prompt Grounding & Safety

- The prompt enforces **strict grounding**  
- Every statement must be supported by retrieved content  
- Citations are displayed clearly to the user  
- If evidence is insufficient, the system responds explicitly instead of guessing  

---

### Language Model

- **Local LLM**: **Phi-3**  
- Chosen for:
  - Lightweight deployment  
  - Strong reasoning when grounded in evidence  
  - Improved safety behaviors  

- Runs locally via Ollama for privacy and reliability.

---

## 🖥️ User Interface (Streamlit)

- Conversational medical assistant experience  
- No raw semantic search exposed to the user  
- Clean clinical response formatting  
- Clearly visible citations  
- Safety disclaimer reminding users to rely on professional medical judgment  

---

## ⏱️ Performance Goals

- **Target response time**: under **3 seconds**  
- Designed for real-world clinical workflows  
- Optimized retrieval and lightweight inference  

---

## 📊 Evaluation & Metrics (Planned)

Automated evaluation metrics are not yet implemented but are a **near-term milestone**.

Planned metrics include:
- Retrieval relevance (Recall@K)  
- Citation coverage  
- Grounding accuracy  
- Hallucination rate  
- Response latency  
- User trust and adoption signals  

These metrics will continuously validate clinical reliability and system safety.

---

## 🔐 Privacy & Security

- Private PDFs are processed locally  
- No patient data is sent to external APIs  
- Vectorized content remains within the controlled environment  
- Designed to align with GDPR and medical confidentiality expectations  

---

## ⚙️ Installation & Execution

This project is designed to run **locally**, ensuring privacy, low latency, and full control over medical data.

---

### 🧩 Prerequisites

Make sure the following are installed on your system:

- **Python 3.10+**
- **pip** (Python package manager)
- **Git**
- **Virtual environment tool** (`venv` recommended)
- **Ollama** (for local LLM inference)

---

### 📥 Clone the Repository

git clone https://github.com/IbtissamLou/Medical-assistant-RAG.git
cd medical-rag-assistant

---

### 🤖 Install & Run Local LLM (Ollama)

- Start Ollama in a separate terminal: ollama serve
- Pull the recommended lightweight medical-safe model: ollama pull phi3
- You can verify installed models with: ollama list

---

### ▶️ Run the Application

streamlit run main.py


## 🔥 Future Improvements

Planned high-impact extensions include:

### 🧩 Multi-Agent Medical Assistant
- Specialized agents for:
  - Literature retrieval  
  - Safety validation  
  - Clinical summarization  
  - Evidence cross-checking  

### 🩻 Medical Image Understanding
- Analysis of:
  - X-rays  
  - CT scans  
  - Dermatology images  
  - Ultrasound frames  

- Combined with literature retrieval for explainable findings.

### 📚 Expanded Medical Data Sources
- WHO  
- CDC  
- ClinicalTrials.gov  
- Clinical guideline repositories  
- ICD-10 and SNOMED integration  

### ⚡ Performance & Scalability
- Model quantization  
- Smart caching  
- Hybrid local/cloud inference  
- Improved response time under load  

---

## 🏁 Vision

By combining **robust retrieval**, **transparent sourcing**, and a **safety-first design**, this project aims to become a **trusted, explainable AI partner** for healthcare professionals and medical learners.

Continuous improvement will be driven by measurable outcomes, ensuring that the assistant remains accurate, auditable, and clinically responsible.

## 🧑‍💻 Authors

Ibtissam Lou — Data Scientist & ML Engineer - Contact : ibtissamloukili20@gmail.com