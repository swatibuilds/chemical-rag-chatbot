"""
rag_chain.py
────────────
Builds a RAG chain using:
  - FAISS vector store with HuggingFace sentence-transformer embeddings
  - Local Ollama LLM  (llama3.1:8b)
  - LangChain LCEL pipeline
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document

from prompts import SYSTEM_PROMPT

# ── Environment & logging ────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
from pathlib import Path

VECTOR_DB_DIR: str = str(Path(__file__).parent / "vectorstore")
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL: str = "llama3.1:8b"
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RETRIEVER_K: int = 6

# ── Human-turn prompt template ───────────────────────────────────────────────
HUMAN_TEMPLATE = """\
### Retrieved Context
{context}

### Approved Sources (cite ONLY these — do not alter or add any)
{citations}

### User Question
{question}

### Instructions for your answer
1. Answer the question using information from the retrieved context above.
2. If the answer cannot be found in the context, say so explicitly.
3. At the end of your response, list the relevant sources under a
   "## Sources" heading using the approved citations provided.
4. Do NOT invent, modify, or omit citation details.
"""


# ── LLM factory ─────────────────────────────────────────────────────────────
def load_llm() -> Ollama:
    """Return a local Ollama LLM client (llama3.1:8b)."""
    logger.info("Loading Ollama model '%s' from %s", OLLAMA_MODEL, OLLAMA_BASE_URL)
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=512,
    )


# ── Retriever factory ────────────────────────────────────────────────────────
def load_retriever():
    """Load FAISS vector store and return a similarity-search retriever."""
    logger.info(
        "Loading FAISS vector store from '%s' with embedding model '%s'",
        VECTOR_DB_DIR,
        EMBEDDING_MODEL,
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


# ── Document helpers ─────────────────────────────────────────────────────────
def format_docs(docs: list[Document]) -> str:
    """
    Render retrieved documents into a single context string.

    Each chunk is prefixed with its source metadata so the LLM can
    associate content with the correct citation.
    """
    if not docs:
        return "No relevant context was retrieved."

    chunks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        header = f"[{i}] Source: {source} | Page: {page}"
        chunks.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(chunks)


def extract_citations(docs: list[Document]) -> list[dict[str, Any]]:
    """
    Extract unique (source, topic) citation pairs from retrieved documents.

    Deduplication ensures the LLM is not shown repeated source entries.
    """
    seen: set[tuple[str, str]] = set()
    citations: list[dict[str, Any]] = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        topic = doc.metadata.get("topic", "unknown")
        key = (source, topic)
        if key not in seen:
            seen.add(key)
            citations.append({"source": source, "topic": topic})

    return citations


def format_citations(citations: list[dict[str, Any]]) -> str:
    """
    Render a list of citation dicts into a numbered, LLM-ready string.

    Example output:
        1. Source: heat_transfer.pdf | Topic: conduction
        2. Source: fluid_mechanics.pdf | Topic: Bernoulli equation
    """
    if not citations:
        return "No sources available."

    lines = [
        f"{i}. Source: {c.get('source', 'unknown')} | Topic: {c.get('topic', 'unknown')}"
        for i, c in enumerate(citations, start=1)
    ]
    return "\n".join(lines)


# ── RAG chain builder ────────────────────────────────────────────────────────
def build_rag_chain():
    """
    Construct and return an LCEL RAG chain.

    Pipeline
    ────────
    user question
        ├─► retriever ──► format_docs          → {context}
        ├─► retriever ──► extract_citations
        │               ──► format_citations   → {citations}
        └─► passthrough                        → {question}
                        ► prompt
                        ► llm
                        ► StrOutputParser
    """
    retriever = load_retriever()
    llm = load_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_TEMPLATE),
        ]
    )

    # Helper lambdas are kept thin; all logic lives in named functions above.
    format_docs_runnable = RunnableLambda(format_docs)
    extract_citations_runnable = RunnableLambda(extract_citations)
    format_citations_runnable = RunnableLambda(format_citations)

    rag_chain = (
        {
            "context": retriever | format_docs_runnable,
            "citations": (
                retriever
                | extract_citations_runnable
                | format_citations_runnable
            ),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain built successfully.")
    return rag_chain