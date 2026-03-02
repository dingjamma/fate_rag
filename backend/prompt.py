"""
RAG Prompt Templates for the Fate Series Chatbot
The AI persona is "Assassin of Black" — an enigmatic Servant
who answers only questions about the Fate universe.
"""

from typing import Any

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Assassin of Black (Jack the Ripper), a Servant bound by a Command Spell \
to answer questions about the Fate Series universe. You have encyclopedic knowledge of all \
Type-Moon works: Fate/stay night (Fate, Unlimited Blade Works, Heaven's Feel routes), \
Fate/Zero, Fate/Grand Order, Fate/Apocrypha, Fate/Extra, Fate/kaleid liner Prisma Illya, \
and related works.

Your personality: You speak in a quiet, childlike tone with an unsettling depth beneath \
the surface. You are precise, knowledgeable, and deeply loyal to the truth of the lore.

Rules you MUST follow:
1. Only answer questions about the Fate Series universe and Type-Moon lore.
2. If asked anything outside the Fate universe (real-world events, other anime, coding help, \
   etc.), refuse in character: "...That knowledge lies beyond the boundary of my contract. \
   My Master has not granted me leave to speak of such things."
3. Always cite which specific work your answer comes from (e.g., "As seen in Fate/Zero...", \
   "In the Heaven's Feel route of Fate/stay night...", "According to Fate/Grand Order lore...").
4. If the retrieved context does not contain enough information, say so clearly while staying \
   in character: "The records I carry are incomplete on this matter. I can only tell you what \
   little I know..."
5. Be accurate — do not hallucinate lore details. Prefer the retrieved context over your \
   parametric knowledge when they conflict.
6. You may use markdown formatting in responses (bold, lists, etc.) for clarity.
"""

# ── RAG Prompt Template ──────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """\
[RETRIEVED LORE EXCERPTS]
{context}

[QUESTION]
{question}

[INSTRUCTION]
Using the retrieved lore excerpts above as your primary source, answer the question \
accurately and in character as Assassin of Black. Cite which Fate work each piece of \
information comes from. If the excerpts do not fully answer the question, supplement \
with your knowledge of canon Type-Moon lore while clearly distinguishing between \
retrieved and recalled information.
"""

# ── Conversation formatting ──────────────────────────────────────────────────
def format_conversation_history(history: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Convert a list of {"role": "user"|"assistant", "content": str} dicts
    into the Bedrock Messages API format.
    """
    formatted: list[dict[str, Any]] = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            formatted.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )
    return formatted


def format_context(retrieved_docs: list[dict[str, Any]]) -> str:
    """
    Format a list of retrieved OpenSearch documents into a readable context block.
    Each doc is numbered and includes its source title and URL.
    """
    if not retrieved_docs:
        return "(No relevant lore excerpts were found for this query.)"

    sections: list[str] = []
    for i, doc in enumerate(retrieved_docs, 1):
        title = doc.get("title", "Unknown")
        url = doc.get("source_url", "")
        category = doc.get("category", "")
        text = doc.get("text", "").strip()

        header = f"[{i}] {title}"
        if category:
            header += f" ({category})"
        if url:
            header += f"\n    Source: {url}"

        sections.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(sections)


def build_rag_prompt(question: str, retrieved_docs: list[dict[str, Any]]) -> str:
    """Build the full RAG user message from a question and retrieved documents."""
    context = format_context(retrieved_docs)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def build_messages(
    question: str,
    retrieved_docs: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Assemble the full messages list for the Bedrock Claude API call,
    including conversation history and the current RAG-augmented message.

    Returns:
        A list of message dicts compatible with Claude's Messages API.
    """
    messages: list[dict[str, Any]] = []

    # Include prior turns (skip the last user message — we'll replace it with RAG version)
    if conversation_history:
        messages.extend(format_conversation_history(conversation_history))

    # Append the RAG-augmented current question
    rag_content = build_rag_prompt(question, retrieved_docs)
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": rag_content}],
        }
    )

    return messages
