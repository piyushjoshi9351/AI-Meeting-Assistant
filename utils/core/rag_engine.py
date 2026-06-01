import os
import logging

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------

def get_llm() -> ChatMistralAI:

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not found."
        )

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
    )

# -----------------------------------------------------------------------------
# Context Formatter
# -----------------------------------------------------------------------------

def format_docs(docs) -> str:
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

def get_prompt():

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert AI Meeting Assistant.

Answer ONLY using the provided meeting transcript context.

Rules:
- Do not hallucinate.
- If answer is unavailable, say:
  "I could not find this information in the meeting transcript."
- Keep answers concise.
- Mention speaker names if available.
- Mention deadlines and owners when relevant.

Meeting Context:
{context}
"""
            ),
            ("human", "{question}")
        ]
    )

# -----------------------------------------------------------------------------
# Build New RAG Chain
# -----------------------------------------------------------------------------

def build_rag_chain(
    transcript: str
):

    logger.info(
        "Building vector database..."
    )

    vector_store = build_vector_store(
        transcript
    )

    retriever = get_retriever(
        vector_store,
        k=4
    )

    llm = get_llm()

    rag_chain = (
        {
            "context":
                retriever
                | RunnableLambda(format_docs),

            "question":
                RunnablePassthrough(),
        }
        | get_prompt()
        | llm
        | StrOutputParser()
    )

    logger.info(
        "RAG chain ready."
    )

    return rag_chain

# -----------------------------------------------------------------------------
# Load Existing RAG Chain
# -----------------------------------------------------------------------------

def load_rag_chain():

    logger.info(
        "Loading existing vector database..."
    )

    vector_store = load_vector_store()

    retriever = get_retriever(
        vector_store,
        k=4
    )

    llm = get_llm()

    rag_chain = (
        {
            "context":
                retriever
                | RunnableLambda(format_docs),

            "question":
                RunnablePassthrough(),
        }
        | get_prompt()
        | llm
        | StrOutputParser()
    )

    logger.info(
        "Existing RAG chain loaded."
    )

    return rag_chain

# -----------------------------------------------------------------------------
# Ask Question
# -----------------------------------------------------------------------------

def ask_question(
    rag_chain,
    question: str
) -> str:

    logger.info(
        f"Question: {question}"
    )

    answer = rag_chain.invoke(
        question
    )

    logger.info(
        "Answer generated."
    )

    return answer