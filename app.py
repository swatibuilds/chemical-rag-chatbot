"""
app.py
──────
Streamlit front-end for the Chemical Engineering RAG chatbot.
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chatbot_backend import chatbot

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ChemBot · Chemical Engineering Assistant",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080c10 !important;
    color: #c9d1d9;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

/* ── App shell ── */
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
section.main > div.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1c2333 !important;
    padding: 1.5rem 1.2rem !important;
    width: 280px !important;
}
section[data-testid="stSidebar"] * { color: #8b949e !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #1c2333 !important;
    margin: 1rem 0 !important;
}
section[data-testid="stSidebar"] p {
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
}

/* Sidebar new chat button */
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid #238636 !important;
    color: #3fb950 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    cursor: pointer !important;
    transition: background 0.2s, color 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #238636 !important;
    color: #ffffff !important;
}

/* ── Main content area ── */
.chembot-wrapper {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #080c10;
}

/* ── Top header bar ── */
.chembot-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 1rem 2rem;
    border-bottom: 1px solid #1c2333;
    background: #0d1117;
    flex-shrink: 0;
}
.chembot-header-icon {
    font-size: 1.4rem;
    line-height: 1;
}
.chembot-header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: #e6edf3;
    letter-spacing: 0.04em;
}
.chembot-header-subtitle {
    font-size: 0.75rem;
    color: #484f58;
    margin-left: auto;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #3fb950;
    display: inline-block;
    margin-right: 0.4rem;
    box-shadow: 0 0 6px #3fb950;
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Scrollable chat feed ── */
.chembot-messages {
    flex: 1;
    overflow-y: auto;
    padding: 2rem 2rem 1rem 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    scrollbar-width: thin;
    scrollbar-color: #1c2333 transparent;
}
.chembot-messages::-webkit-scrollbar { width: 4px; }
.chembot-messages::-webkit-scrollbar-track { background: transparent; }
.chembot-messages::-webkit-scrollbar-thumb { background: #1c2333; border-radius: 4px; }

/* ── Message rows ── */
.msg-row {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    animation: fadeUp 0.28s ease both;
}
.msg-row.user { flex-direction: row-reverse; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Avatars ── */
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
}
.avatar.user-avatar {
    background: #1f3040;
    color: #58a6ff;
    border: 1px solid #1c3a54;
}
.avatar.bot-avatar {
    background: #0f2318;
    color: #3fb950;
    border: 1px solid #1a3c24;
}

/* ── Bubble bodies ── */
.bubble {
    max-width: min(680px, 72%);
    padding: 0.8rem 1.1rem;
    border-radius: 10px;
    font-size: 0.9rem;
    line-height: 1.65;
    word-break: break-word;
}
.bubble.user-bubble {
    background: #161d27;
    border: 1px solid #1c2d40;
    color: #cdd9e5;
    border-radius: 10px 4px 10px 10px;
}
.bubble.bot-bubble {
    background: #0d1117;
    border: 1px solid #1c2333;
    color: #c9d1d9;
    border-radius: 4px 10px 10px 10px;
}

/* ── Inline code & pre in bubbles ── */
.bubble code {
    font-family: 'IBM Plex Mono', monospace;
    background: #161b22;
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.83em;
    color: #79c0ff;
    border: 1px solid #21262d;
}
.bubble pre {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
    margin-top: 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82em;
    color: #adbac7;
}

/* ── Sources section in bot bubble ── */
.bubble .sources-block {
    margin-top: 0.75rem;
    padding-top: 0.6rem;
    border-top: 1px solid #1c2333;
    font-size: 0.8rem;
    color: #484f58;
    font-family: 'IBM Plex Mono', monospace;
}
.bubble .sources-block span { color: #3fb950; }

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    gap: 1rem;
    padding: 3rem;
    opacity: 0.6;
}
.empty-icon {
    font-size: 3rem;
    line-height: 1;
}
.empty-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    color: #8b949e;
    letter-spacing: 0.06em;
}
.empty-hint {
    font-size: 0.82rem;
    color: #484f58;
    text-align: center;
    max-width: 420px;
    line-height: 1.7;
}

/* ── Suggested prompts ── */
.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    max-width: 560px;
}
.suggestion-chip {
    background: #0d1117;
    border: 1px solid #1c2333;
    color: #8b949e;
    font-size: 0.78rem;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    cursor: pointer;
    transition: border-color 0.18s, color 0.18s;
    font-family: 'IBM Plex Sans', sans-serif;
}
.suggestion-chip:hover {
    border-color: #3fb950;
    color: #3fb950;
}

/* ── Input bar ── */
.chembot-input-bar {
    padding: 1rem 2rem 1.4rem 2rem;
    background: #0d1117;
    border-top: 1px solid #1c2333;
    flex-shrink: 0;
}

/* Streamlit chat_input overrides */
[data-testid="stChatInput"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    box-shadow: 0 0 0 0px #3fb950 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3fb950 !important;
    box-shadow: 0 0 0 3px rgba(63,185,80,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    caret-color: #3fb950 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #484f58 !important;
}
[data-testid="stChatInputSubmitButton"] svg { stroke: #3fb950 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #3fb950 !important; }
[data-testid="stSpinner"] > div { border-top-color: #3fb950 !important; }

/* ── Thinking indicator ── */
.thinking-dots span {
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #3fb950;
    margin: 0 2px;
    animation: blink 1.2s infinite both;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.15; transform: scale(0.85); }
    40% { opacity: 1; transform: scale(1.1); }
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "message_history" not in st.session_state:
    st.session_state.message_history: list[dict] = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id: str = str(uuid.uuid4())


def reset_conversation() -> None:
    st.session_state.message_history = []
    st.session_state.thread_id = str(uuid.uuid4())


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚗️ ChemBot")
    st.markdown("---")
    st.markdown(
        "**Chemical Engineering**  \nKnowledge Assistant  \n\n"
        "Ask about reactions, unit operations, process safety, "
        "thermodynamics, transport phenomena, and more."
    )
    st.markdown("---")

    if st.button("＋  New conversation"):
        reset_conversation()
        st.rerun()

    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("llama3.1:8b · Ollama")
    st.markdown("**Embeddings**")
    st.markdown("all-MiniLM-L6-v2")
    st.markdown("**Retrieval**")
    st.markdown("FAISS · top-6 chunks")
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.72rem; color:#30363d;'>Session · "
        f"`{st.session_state.thread_id[:8]}…`</span>",
        unsafe_allow_html=True,
    )


# ── Header bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chembot-header">
    <div class="chembot-header-icon">⚗️</div>
    <div class="chembot-header-title">ChemBot</div>
    <div class="chembot-header-subtitle">
        <span class="status-dot"></span>llama3.1:8b · local
    </div>
</div>
""", unsafe_allow_html=True)


# ── Chat messages ─────────────────────────────────────────────────────────────
SUGGESTED = [
    "What is the Arrhenius equation?",
    "Explain Fick's law of diffusion",
    "How does a distillation column work?",
    "What is HAZOP analysis?",
]

chat_area = st.container()

with chat_area:
    if not st.session_state.message_history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⚗️</div>
            <div class="empty-title">CHEMICAL ENGINEERING ASSISTANT</div>
            <div class="empty-hint">
                Ask questions about reactions, unit operations, process safety,
                thermodynamics, or transport phenomena — grounded in your document corpus.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggestion chips via columns
        cols = st.columns(len(SUGGESTED))
        for col, prompt in zip(cols, SUGGESTED):
            with col:
                if st.button(prompt, key=f"suggest_{prompt[:12]}"):
                    st.session_state._pending_input = prompt
    else:
        st.markdown('<div class="chembot-messages">', unsafe_allow_html=True)
        for msg in st.session_state.message_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-row user">
                    <div class="avatar user-avatar">U</div>
                    <div class="bubble user-bubble">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-row">
                    <div class="avatar bot-avatar">⚗</div>
                    <div class="bubble bot-bubble">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── Input bar ─────────────────────────────────────────────────────────────────
# Handle suggestion chip clicks
pending = st.session_state.pop("_pending_input", None)
user_input: str | None = st.chat_input("Ask a chemical engineering question…") or pending

if user_input:
    user_input = user_input.strip()
    st.session_state.message_history.append({"role": "user", "content": user_input})

    # Stream assistant response
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.spinner(""):
        st.markdown("""
        <div class="msg-row" style="padding: 0 2rem;">
            <div class="avatar bot-avatar">⚗</div>
            <div class="bubble bot-bubble">
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        def _ai_stream():
            for chunk, _meta in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        ai_response = st.write_stream(_ai_stream())

    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_response}
    )
    st.rerun()