# ⚗️ ChemBot — Chemical Engineering RAG Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Graph-FF6B35?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-llama3.1%3A8b-black?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-009688?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

**A fully local, production-grade Retrieval-Augmented Generation (RAG) chatbot for Chemical Engineering knowledge — featuring multi-turn memory, automatic query refinement, and a polished dark-mode UI.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Pipeline Deep Dive](#-pipeline-deep-dive)
- [Setup & Installation](#-setup--installation)
- [Running the App](#-running-the-app)
- [Demo Query Sequence](#-demo-query-sequence)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

ChemBot is a fully offline RAG chatbot that answers Chemical Engineering questions grounded strictly in your own document corpus. It never hallucinates — if the answer isn't in the documents, it says so.

Built on **LangGraph** for stateful multi-turn conversation, **FAISS** for fast semantic retrieval, and **Ollama (llama3.1:8b)** as the local LLM — no API keys, no cloud, no cost per query.

---

Demo video Link: https://youtu.be/559nvfMSS6Q

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  PDF Documents (by topic folder)                                 │
│       │                                                          │
│       ▼                                                          │
│  PyPDFLoader  ──►  RecursiveCharacterTextSplitter                │
│  (+ topic & source metadata)   (chunk=800, overlap=100)          │
│       │                                                          │
│       ▼                                                          │
│  HuggingFace Embeddings  ──►  FAISS Index  ──►  Saved to disk   │
│  (all-MiniLM-L6-v2)           (vectorstore/)                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                          │
│                                                                  │
│  User Message                                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────┐                                         │
│  │  query_refiner_node │  ◄── resolves ambiguous / context-     │
│  │  (llama3.1:8b)      │       dependent queries using history  │
│  └─────────┬───────────┘                                         │
│            │  refined_query                                      │
│            ▼                                                     │
│  ┌─────────────────────┐                                         │
│  │     chat_node       │                                         │
│  │                     │                                         │
│  │  FAISS Retriever    │  ◄── top-6 similarity chunks           │
│  │       │             │                                         │
│  │  format_docs        │  ◄── numbered context blocks           │
│  │  extract_citations  │  ◄── deduplicated source metadata      │
│  │  format_citations   │                                         │
│  │       │             │                                         │
│  │  ChatPromptTemplate │  ◄── system prompt + context +         │
│  │       │             │       citations + refined question      │
│  │  llama3.1:8b (local)│                                         │
│  │       │             │                                         │
│  │  StrOutputParser    │                                         │
│  └─────────┬───────────┘                                         │
│            │  AI Response (streamed)                             │
│            ▼                                                     │
│       Streamlit UI  ──►  InMemorySaver (thread history)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔁 Automatic Query Refinement
Before any retrieval happens, a dedicated **query refiner node** inspects the full conversation history and rewrites ambiguous or context-dependent questions into fully self-contained search queries.

| User says | History contains | Refined query sent to retriever |
|-----------|-----------------|----------------------------------|
| `"explain that further"` | Arrhenius equation | `"Explain the Arrhenius equation in more detail."` |
| `"what are its applications?"` | distillation → heat exchangers → **Fick's law** | `"What are the applications of Fick's law of diffusion?"` |
| `"give me an example"` | CSTR → PFR → **Ergun equation** | `"Give a real-world example of the Ergun equation."` |

The refiner always resolves to the **most recently discussed concept** when history contains multiple topics — preventing stale or merged retrievals.

### 🧠 Multi-turn Memory with LangGraph
Conversation state is managed by a **LangGraph `StateGraph`** with `InMemorySaver` checkpointing. Every session gets a unique `thread_id` so multiple users or sessions never bleed into each other.

### 📄 Grounded Retrieval with Citations
Every answer is backed by retrieved document chunks. The RAG chain:
- Retrieves the **top-6 most semantically similar chunks** from the FAISS index
- Attaches **source filename and topic** metadata to every chunk
- Deduplicates citations before passing them to the LLM
- Instructs the LLM to list sources under a `## Sources` section

### 🗂 Topic-aware Document Ingestion
Documents are organised into topic sub-folders. Each page is tagged with:
- `source` — the PDF filename
- `topic` — the sub-folder name (e.g. `Chemical Reactions`, `unit_operations`)

This metadata flows all the way through to the final citation block.

### 🖥 Polished Dark-mode UI
Built in Streamlit with a fully custom CSS layer:
- IBM Plex Sans + IBM Plex Mono typography
- GitHub-inspired deep navy dark theme
- Animated message bubbles with user/bot avatars
- Live streaming token output via `st.write_stream`
- Pulsing thinking indicator while the model generates
- Suggested prompt chips on the empty state screen
- New conversation button with isolated thread ID

---

## 📁 Project Structure

```
chemical-rag-chatbot/
│
├── app.py                  # Streamlit UI — chat interface
├── chatbot_backend.py      # LangGraph graph — query refiner + chat nodes
├── rag_chain.py            # LCEL RAG chain — retriever, prompt, LLM
├── prompts.py              # System prompt for the Chemical Engineering LLM
├── ingest.py               # Ingestion pipeline — PDF → FAISS index
│
├── Documents/              # Your PDF corpus (not committed)
│   ├── Chemical Reactions/
│   ├── unit_operations/
│   ├── process_safety/
│   ├── equipment_basics/
│   └── MS-DS/
│
├── vectorstore/            # FAISS index (generated by ingest.py, not committed)
│   ├── index.faiss
│   └── index.pkl
│
├── .env                    # Your secrets (not committed)
├── .env.example            # Template for required env vars
├── requirements.txt
└── README.md
```

---

## 🔬 Pipeline Deep Dive

### 1 — Ingestion (`ingest.py`)

```
Documents/
  └── <topic_folder>/
        └── *.pdf
              │
              ▼
        PyPDFLoader
        page.metadata["topic"] = folder name
        page.metadata["source"] = filename
              │
              ▼
        RecursiveCharacterTextSplitter
        chunk_size=800, chunk_overlap=100
              │
              ▼
        HuggingFaceEmbeddings (all-MiniLM-L6-v2)
              │
              ▼
        FAISS.from_documents()
              │
              ▼
        vectorstore/  (saved to disk)
```

Run once before starting the app:
```bash
python ingest.py
```

Optional CLI flags:
```bash
python ingest.py --data-dir /path/to/docs --chunk-size 1000 --chunk-overlap 150
```

---

### 2 — Query Refinement (`chatbot_backend.py`)

```
Full message history
        +
Latest user message
        │
        ▼
  _REFINER_SYSTEM prompt
  (rules + recency rule + resolution guide)
        │
        ▼
  llama3.1:8b (local Ollama)
        │
        ▼
  refined_query  ──►  stored in ChatState
```

The refiner fires on **every turn**. On the first turn (empty history) it short-circuits and passes the query through unchanged — zero extra LLM cost.

---

### 3 — RAG Chain (`rag_chain.py`)

```
refined_query
      │
      ├──► FAISS retriever (top-6 chunks)
      │           │
      │     ┌─────┴──────────────────┐
      │     │                        │
      │  format_docs          extract_citations
      │  (numbered blocks     (deduped source
      │   with metadata)       + topic pairs)
      │     │                        │
      │     │                 format_citations
      │     │                 (numbered list)
      │     └─────┬──────────────────┘
      │           │
      ▼           ▼
  ChatPromptTemplate
  {system} + {context} + {citations} + {question}
      │
      ▼
  llama3.1:8b
      │
      ▼
  StrOutputParser  ──►  streamed to UI
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- llama3.1:8b pulled: `ollama pull llama3.1:8b`

### 1 — Clone the repo
```bash
git clone https://github.com/swatibuilds/chemical-rag-chatbot.git
cd chemical-rag-chatbot
```

### 2 — Create and activate virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Configure environment
```bash
cp .env.example .env
# Edit .env and set VECTOR_DB_DIR if needed
```

### 5 — Add your documents
Place PDF files inside `Documents/` organised by topic sub-folder:
```
Documents/
  ├── Chemical Reactions/   ← PDFs go here
  ├── unit_operations/
  └── process_safety/
```

### 6 — Run ingestion
```bash
python ingest.py
```
This generates the `vectorstore/` directory. Run this once, or whenever you add new documents.

---

## 🚀 Running the App

Make sure Ollama is running in the background:
```bash
ollama serve
```

Then launch the Streamlit app:
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎬 Demo Query Sequence

Use this sequence to see every feature fire in order:

| # | Query | What it demonstrates |
|---|-------|----------------------|
| 1 | `"What is the Arrhenius equation and how does temperature affect reaction rate?"` | Core RAG retrieval + citations |
| 2 | `"Can you give me a real world example of that?"` | Refiner resolves `"that"` → Arrhenius equation |
| 3 | `"Explain the working principle of a distillation column and what factors affect its efficiency?"` | Cross-domain retrieval (topic switch) |
| 4 | `"What are its industrial applications?"` | Refiner applies recency rule → resolves to distillation, not Arrhenius |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Ollama · llama3.1:8b (fully local) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| RAG Framework | LangChain LCEL |
| Conversation Graph | LangGraph `StateGraph` + `InMemorySaver` |
| Document Loader | LangChain `PyPDFLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| UI | Streamlit + custom CSS |
| Environment | `python-dotenv` |

---

<div align="center">
<sub>Built with LangChain · LangGraph · Ollama · FAISS · Streamlit</sub>
</div>
