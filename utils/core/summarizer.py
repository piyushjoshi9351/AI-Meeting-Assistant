import os
import logging
from typing import List

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# LLM Loader
# -----------------------------------------------------------------------------

def get_llm() -> ChatMistralAI:
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not found in environment variables."
        )

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
    )


# -----------------------------------------------------------------------------
# Transcript Splitter
# -----------------------------------------------------------------------------

def split_transcript(transcript: str) -> List[str]:
    """
    Split large transcripts into manageable chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


# -----------------------------------------------------------------------------
# Meeting Summary
# -----------------------------------------------------------------------------

def summarize(transcript: str) -> str:
    """
    Generate a professional meeting summary
    using Map-Reduce summarization.
    """

    logger.info("Generating summary...")

    llm = get_llm()

    chunks = split_transcript(transcript)

    logger.info(f"Transcript split into {len(chunks)} chunk(s).")

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this section of a meeting transcript. "
                "Focus on key discussions, decisions, and outcomes."
            ),
            ("human", "{text}")
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = []

    for idx, chunk in enumerate(chunks, start=1):
        logger.info(
            f"Summarizing chunk {idx}/{len(chunks)}"
        )

        summary = map_chain.invoke(
            {"text": chunk}
        )

        chunk_summaries.append(summary)

    combined_text = "\n\n".join(chunk_summaries)

    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert meeting analyst.

Create a final report with:

1. Meeting Overview
2. Key Discussion Points
3. Important Decisions
4. Risks / Concerns
5. Next Steps

Use professional bullet points.
"""
            ),
            ("human", "{text}")
        ]
    )

    reduce_chain = (
        reduce_prompt
        | llm
        | StrOutputParser()
    )

    final_summary = reduce_chain.invoke(
        {"text": combined_text}
    )

    logger.info("Summary generated successfully.")

    return final_summary


# -----------------------------------------------------------------------------
# Title Generator
# -----------------------------------------------------------------------------

def generate_title(transcript: str) -> str:
    """
    Generate a short professional meeting title.
    """

    logger.info("Generating title...")

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Generate a professional meeting title.

Rules:
- Maximum 8 words
- No quotation marks
- No explanation
- Return title only
"""
            ),
            ("human", "{text}")
        ]
    )

    chain = (
        title_prompt
        | llm
        | StrOutputParser()
    )

    title = chain.invoke(
        {"text": transcript[:2000]}
    )

    return title.strip()