"""
chatbot_backend.py
──────────────────
LangGraph-powered RAG chatbot backend for the Chemical Engineering assistant.

Pipeline per turn:
    START
      └─► query_refiner_node   — resolves ambiguous / context-dependent queries
            └─► chat_node      — retrieves context, generates grounded answer
                  └─► END

Query refinement example:
    History : "What is the Arrhenius equation?"
    User    : "Explain that in more detail."
    Refined : "Explain the Arrhenius equation in more detail."
"""

from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from rag_chain_hf import build_rag_chain, load_llm

# ── Environment & logging ─────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Shared singletons (loaded once at startup) ────────────────────────────────
logger.info("Initialising LLM…")
llm = load_llm()

logger.info("Building RAG chain…")
rag = build_rag_chain()


# ── Graph state ───────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    refined_query: str          # set by query_refiner_node, consumed by chat_node


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_history_string(messages: list[BaseMessage], *, exclude_last: bool = True) -> str:
    """
    Render the message list as a plain-text dialogue string.

    Args:
        messages:     Full message history from the graph state.
        exclude_last: When True, omit the final message (the current user query).

    Returns:
        Multi-line string of the form "User: …\\nAssistant: …\\n …"
        Empty string if there is no prior history.
    """
    target = messages[:-1] if exclude_last else messages
    lines: list[str] = []
    for msg in target:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content.strip()}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content.strip()}")
    return "\n".join(lines)


def _latest_human_query(messages: list[BaseMessage]) -> str:
    """Return the content of the most recent HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    raise ValueError("No HumanMessage found in state.")


# ── Query refiner prompt ──────────────────────────────────────────────────────
_REFINER_SYSTEM = """\
You are a query-rewriting assistant for a Chemical Engineering RAG system.

Your only job is to rewrite the user's LATEST question into a fully \
self-contained, unambiguous search query — suitable for semantic retrieval \
over a document corpus — by resolving any pronouns, demonstratives, or \
implicit references using the conversation history.

═══════════════════════════════════════════════
CORE RULES
═══════════════════════════════════════════════
1. Return ONLY the rewritten query — no preamble, explanation, labels, or quotes.
2. If the query is already fully self-contained and unambiguous, return it unchanged.
3. Never answer the question — only rewrite it.
4. Keep the rewrite concise (one sentence preferred; two sentences maximum).
5. Preserve the user's original intent exactly — do not add assumptions or extra scope.
6. The rewritten query must make complete sense with ZERO conversation context.

═══════════════════════════════════════════════
RECENCY RULE  (most important for multi-topic histories)
═══════════════════════════════════════════════
When the conversation history contains multiple concepts and the user asks a
vague follow-up ("explain it", "what's its use?", "give an example", "how does
it work?", "tell me more"), ALWAYS resolve the reference to the SINGLE MOST
RECENTLY DISCUSSED concept — never combine multiple topics unless the user
explicitly names more than one in their latest message.

  History  : Arrhenius equation → distillation → Fick's law of diffusion
  User     : "give me an example"
  Correct  : "Give an example of Fick's law of diffusion."
  WRONG    : "Give an example of the Arrhenius equation, distillation, and Fick's law."

═══════════════════════════════════════════════
REFERENCE RESOLUTION GUIDE
═══════════════════════════════════════════════
Resolve these patterns against the most recent relevant concept:

  Pronouns / demonstratives
    "it", "this", "that", "they", "these", "those"
    → replace with the noun the pronoun refers to

  Vague follow-ups
    "explain further", "tell me more", "elaborate", "go deeper"
    → "Explain <latest concept> in more detail."

  Usage / application questions
    "what is its use?", "where is it applied?", "what are its applications?"
    → "What are the applications of <latest concept>?"

  Example requests
    "give an example", "show me a case", "real-world example"
    → "Give a real-world example of <latest concept>."

  Comparison follow-ups  (only when user names both explicitly)
    "which is better?", "compare them", "what's the difference?"
    → "Compare <concept A> and <concept B> in the context of <domain>."
    If only one concept is recent and the other is vague ("compare it to
    the previous one"), resolve both from history.

  Definition / mechanism questions
    "how does it work?", "what is it?", "what does it mean?"
    → "How does <latest concept> work?" / "What is <latest concept>?"

  Equation / formula requests
    "write the equation", "give the formula", "mathematical form"
    → "What is the mathematical equation / formula for <latest concept>?"

═══════════════════════════════════════════════
WHAT NOT TO DO
═══════════════════════════════════════════════
✗ Do not merge multiple historical concepts into one query unless the user
  explicitly references more than one in their current message.
✗ Do not add extra context, caveats, or scope beyond what the user asked.
✗ Do not include phrases like "Based on our conversation…" or "As discussed…"
✗ Do not wrap the output in quotes or add a label like "Rewritten:".
✗ Do not answer the question — only rewrite it.

═══════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════

  -- Single concept, vague follow-up --
  History : "What is the Arrhenius equation?"
  User    : "Can you explain that further?"
  Output  : Can you explain the Arrhenius equation in more detail?

  -- Multi-concept history, resolve to LATEST --
  History : Arrhenius equation → CSTR design → Fick's law of diffusion
  User    : "What are its applications?"
  Output  : What are the applications of Fick's law of diffusion?

  -- Multi-concept history, resolve to LATEST --
  History : distillation → heat exchangers → packed bed reactors
  User    : "give me an example"
  Output  : Give a real-world example of a packed bed reactor.

  -- User explicitly names two concepts --
  History : distillation → heat exchangers
  User    : "which one is more energy intensive?"
  Output  : Between distillation and heat exchangers, which process is more energy intensive?

  -- Already self-contained --
  History : (anything)
  User    : "What is Fick's law of diffusion?"
  Output  : What is Fick's law of diffusion?

  -- Empty history --
  History : (empty)
  User    : "How does a CSTR differ from a PFR?"
  Output  : How does a CSTR differ from a PFR?

  -- Equation request --
  History : "Explain the Ergun equation." → "How is it derived?"
  User    : "write the formula"
  Output  : What is the mathematical formula for the Ergun equation?

  -- Usage after multi-topic chat --
  History : ideal gas law → van der Waals equation → Raoult's law
  User    : "what's its use in industry?"
  Output  : What is the industrial use of Raoult's law?
"""

_refiner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _REFINER_SYSTEM),
        (
            "human",
            "Conversation history:\n{history}\n\nLatest user question:\n{question}",
        ),
    ]
)

_refiner_chain = _refiner_prompt | llm | StrOutputParser()


# ── Graph nodes ───────────────────────────────────────────────────────────────
def query_refiner_node(state: ChatState) -> dict:
    """
    Resolve ambiguous or context-dependent user queries before retrieval.

    Reads the full conversation history and the latest user message, then
    produces a self-contained `refined_query` that is stored in the state
    for `chat_node` to use.
    """
    messages = state["messages"]
    latest_query = _latest_human_query(messages)
    history = _build_history_string(messages, exclude_last=True)

    if not history:
        # First turn — nothing to resolve.
        logger.info("Query refiner: no history, passing query through unchanged.")
        return {"refined_query": latest_query}

    logger.info("Query refiner: resolving query against history…")
    refined = _refiner_chain.invoke(
        {"history": history, "question": latest_query}
    ).strip()

    if refined != latest_query:
        logger.info("  Original : %s", latest_query)
        logger.info("  Refined  : %s", refined)
    else:
        logger.info("  Query was already self-contained — no change.")

    return {"refined_query": refined}


def chat_node(state: ChatState) -> dict:
    """
    Run the RAG chain using the refined query and return the AI response.

    Uses `state["refined_query"]` (set by `query_refiner_node`) as the
    retrieval and generation input, so the LLM always receives a fully
    contextualised question.
    """
    refined_query = state.get("refined_query", "")
    if not refined_query:
        # Fallback safety — should never happen in normal flow.
        refined_query = _latest_human_query(state["messages"])
        logger.warning("chat_node: refined_query was empty, falling back to raw query.")

    logger.info("chat_node: invoking RAG chain with query: %s", refined_query)

    response_text: str = rag.invoke(refined_query)

    return {"messages": [AIMessage(content=response_text)]}


# ── Graph assembly ────────────────────────────────────────────────────────────
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("query_refiner_node", query_refiner_node)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "query_refiner_node")
graph.add_edge("query_refiner_node", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

logger.info("Chatbot graph compiled and ready.")