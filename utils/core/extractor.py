import os
import logging

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        temperature=0.2,
    )


# -----------------------------------------------------------------------------
# Generic Extraction Chain
# -----------------------------------------------------------------------------

def build_chain(system_prompt: str):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}")
        ]
    )

    return prompt | llm | StrOutputParser()


# -----------------------------------------------------------------------------
# Action Items
# -----------------------------------------------------------------------------

def extract_action_items(transcript: str) -> str:
    """
    Extract meeting action items.
    """

    logger.info("Extracting action items...")

    chain = build_chain(
        """
You are an expert meeting analyst.

Extract all action items from the meeting.

For each action item provide:

1. Task
2. Owner
3. Deadline

Rules:
- If owner is unknown write "Not specified"
- If deadline is unknown write "Not specified"
- Use a numbered list
- Return only action items
- If none exist, return:
  No action items found.
"""
    )

    return chain.invoke({"text": transcript})


# -----------------------------------------------------------------------------
# Key Decisions
# -----------------------------------------------------------------------------

def extract_key_decisions(transcript: str) -> str:
    """
    Extract important decisions made.
    """

    logger.info("Extracting key decisions...")

    chain = build_chain(
        """
You are an expert meeting analyst.

Extract all key decisions made during the meeting.

Rules:
- Use a numbered list
- Keep each decision concise
- Include important context if needed
- If no decisions exist return:
  No key decisions found.
"""
    )

    return chain.invoke({"text": transcript})


# -----------------------------------------------------------------------------
# Open Questions
# -----------------------------------------------------------------------------

def extract_questions(transcript: str) -> str:
    """
    Extract unresolved questions and follow-ups.
    """

    logger.info("Extracting open questions...")

    chain = build_chain(
        """
You are an expert meeting analyst.

Extract all unresolved questions,
pending discussions,
and follow-up topics.

Rules:
- Use a numbered list
- Keep each item concise
- If no open questions exist return:
  No open questions found.
"""
    )

    return chain.invoke({"text": transcript})


# -----------------------------------------------------------------------------
# Meeting Insights (Bonus Feature)
# -----------------------------------------------------------------------------

def extract_risks(transcript: str) -> str:
    """
    Extract project risks, blockers and concerns.
    Useful for dashboard analytics.
    """

    logger.info("Extracting risks and blockers...")

    chain = build_chain(
        """
You are an expert project manager.

Identify:
- Risks
- Blockers
- Concerns
- Dependencies

Format as a numbered list.

If none found return:
No risks identified.
"""
    )

    return chain.invoke({"text": transcript})