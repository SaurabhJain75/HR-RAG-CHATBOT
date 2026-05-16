"""
prompts.py
----------
All LLM prompt templates for the HR Policy RAG Chatbot.
Keeping prompts here (not scattered in agents.py or tools.py) means:
  - Easy to tune without touching logic files
  - Single place to review what the LLM is being told
  - Simple A/B testing by swapping templates
"""


# ══════════════════════════════════════════════════════════════════════════════
# System Prompt
# The core identity and behavior rules for the HR assistant.
# Sent once at the start of every conversation.
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful and professional HR Policy Assistant for our organization.

Your job is to answer employee questions accurately based ONLY on the HR policy documents provided to you as context.

## Rules you must always follow:

1. **Stay grounded** — Only answer using the provided context. Never make up policies, numbers, or rules.
2. **Be honest about gaps** — If the context doesn't contain enough information to answer, say so clearly. Do not guess.
3. **Be concise** — Give direct, clear answers. Avoid unnecessary filler.
4. **Cite your source** — Always mention which policy document your answer comes from (e.g. "According to the Leave Policy...").
5. **Stay professional** — You represent the HR department. Be warm but formal.
6. **No personal advice** — If an employee asks about their specific case (e.g. "Can I get an exception?"), direct them to contact HR directly.
7. **No topics outside HR** — If asked about non-HR topics (coding, general knowledge, etc.), politely decline and redirect.

## Your tone:
- Clear and professional
- Empathetic (employees may be stressed about policy questions)
- Never robotic or overly formal
"""


# ══════════════════════════════════════════════════════════════════════════════
# RAG Answer Prompt
# Used in agents.py to ask the LLM to answer using retrieved chunks.
# {context} and {question} are filled in at runtime.
# ══════════════════════════════════════════════════════════════════════════════

RAG_PROMPT_TEMPLATE = """Use the following HR policy excerpts to answer the employee's question.

## HR Policy Context:
{context}

## Employee Question:
{question}

## Instructions:
- Answer based ONLY on the context above.
- If the context fully answers the question, give a clear and complete answer.
- If the context partially answers it, answer what you can and note what is unclear.
- If the context does not contain the answer, respond with the fallback message.
- Always mention the source document name when referencing a policy.
- Format your answer clearly. Use bullet points if listing multiple items.

## Answer:"""


# ══════════════════════════════════════════════════════════════════════════════
# Fallback Prompt
# Used when no relevant chunks are retrieved above the similarity threshold.
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_PROMPT_TEMPLATE = """An employee asked the following question, but no relevant HR policy document was found to answer it:

## Employee Question:
{question}

Respond politely and professionally. Tell the employee:
1. You could not find this information in the current HR policy documents.
2. They should contact the HR department directly for accurate information.
3. Provide the HR contact hint: {hr_contact}

Keep the response brief and empathetic.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Fallback Message (static — used directly when skipping LLM for fallback)
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find relevant information in our HR policy documents to answer your question. "
    "For accurate guidance, please reach out to the HR department directly. "
    "They'll be happy to help!"
)


# ══════════════════════════════════════════════════════════════════════════════
# Conversation Starter (shown in app.py UI on load)
# ══════════════════════════════════════════════════════════════════════════════

WELCOME_MESSAGE = (
    "👋 Hello! I'm your HR Policy Assistant. "
    "I can help you find information about leave policies, benefits, code of conduct, "
    "reimbursements, and more. What would you like to know?"
)


# ══════════════════════════════════════════════════════════════════════════════
# Suggested Questions (shown in app.py as quick-start buttons)
# ══════════════════════════════════════════════════════════════════════════════

SUGGESTED_QUESTIONS = [
    "How many casual leaves am I entitled to per year?",
    "What is the work from home policy?",
    "How do I apply for maternity or paternity leave?",
    "What expenses are covered under travel reimbursement?",
    "What is the notice period for resignation?",
    "How does the performance review process work?",
]


# ══════════════════════════════════════════════════════════════════════════════
# HR Contact (used in FALLBACK_PROMPT_TEMPLATE)
# Update this to match your organization's HR contact info
# ══════════════════════════════════════════════════════════════════════════════

HR_CONTACT_INFO = "hr@yourcompany.com or extension 1001"


# ══════════════════════════════════════════════════════════════════════════════
# Helper: build the RAG prompt with context + question filled in
# ══════════════════════════════════════════════════════════════════════════════

def build_rag_prompt(context: str, question: str) -> str:
    """
    Fill the RAG_PROMPT_TEMPLATE with retrieved context and the user's question.

    Args:
        context:  Concatenated text of retrieved document chunks
        question: The user's original question

    Returns:
        Filled prompt string ready to send to the LLM
    """
    return RAG_PROMPT_TEMPLATE.format(
        context=context.strip(),
        question=question.strip()
    )


def build_fallback_prompt(question: str) -> str:
    """
    Fill the FALLBACK_PROMPT_TEMPLATE when no relevant chunks were found.

    Args:
        question: The user's original question

    Returns:
        Filled fallback prompt string
    """
    return FALLBACK_PROMPT_TEMPLATE.format(
        question=question.strip(),
        hr_contact=HR_CONTACT_INFO
    )


def format_context(chunks: list) -> str:
    """
    Convert a list of RetrievedChunk objects into a single context string
    to inject into the RAG prompt.

    Each chunk is labelled with its source file and page number so the LLM
    can cite them accurately.

    Args:
        chunks: List of RetrievedChunk objects from rag.py

    Returns:
        Formatted context string
    """
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.source_file
        page   = f", page {chunk.page_number}" if chunk.page_number else ""
        section = f" — {chunk.section}" if chunk.section else ""
        header = f"[Excerpt {i} | Source: {source}{page}{section}]"
        formatted.append(f"{header}\n{chunk.content.strip()}")

    return "\n\n---\n\n".join(formatted)
