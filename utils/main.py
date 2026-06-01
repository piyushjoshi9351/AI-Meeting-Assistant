from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load Environment Variables
# -----------------------------------------------------------------------------

load_dotenv()

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
    extract_risks,
)

from core.rag_engine import (
    build_rag_chain,
    ask_question,
)

# -----------------------------------------------------------------------------
# Pipeline Function
# -----------------------------------------------------------------------------

def run_pipeline(source: str, language: str = "english") -> dict:
    """
    Complete AI Meeting Assistant pipeline.
    Returns all generated outputs.
    """

    print("\n🚀 Starting AI Meeting Assistant...\n")

    if language.lower() not in ["english", "hinglish"]:
        language = "english"

    # -------------------------------------------------------------------------
    # Audio Processing
    # -------------------------------------------------------------------------

    print("🎧 Processing audio...")
    chunks = process_input(source)

    # -------------------------------------------------------------------------
    # Transcription
    # -------------------------------------------------------------------------

    print("📝 Generating transcript...")
    transcript = transcribe_all(
        chunks,
        language=language
    )

    print("\n📝 Transcript Preview:")
    print("-" * 80)
    print(transcript[:300] + "..." if len(transcript) > 300 else transcript)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("\n🤖 Generating summary...")
    title = generate_title(transcript)
    summary = summarize(transcript)

    # -------------------------------------------------------------------------
    # Information Extraction
    # -------------------------------------------------------------------------

    print("\n🔍 Extracting insights...")

    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    risks = extract_risks(transcript)

    # -------------------------------------------------------------------------
    # Build RAG Knowledge Base
    # -------------------------------------------------------------------------

    print("\n📚 Building knowledge base...")
    rag_chain = build_rag_chain(transcript)

    # -------------------------------------------------------------------------
    # Return Results
    # -------------------------------------------------------------------------

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "risks": risks,
        "rag_chain": rag_chain,
    }


# -----------------------------------------------------------------------------
# CLI Mode
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    try:

        print("\n" + "=" * 80)
        print("🚀 AI MEETING ASSISTANT")
        print("=" * 80)

        source = input(
            "\nEnter YouTube URL or Local File Path:\n> "
        ).strip()

        if not source:
            raise ValueError("No source provided.")

        language = input(
            "\nLanguage (english/hinglish) [default: english]:\n> "
        ).strip().lower()

        if not language:
            language = "english"

        result = run_pipeline(
            source=source,
            language=language
        )

        print("\n" + "=" * 80)
        print(f"📌 TITLE: {result['title']}")
        print("=" * 80)

        print("\n📋 SUMMARY")
        print("-" * 80)
        print(result["summary"])

        print("\n✅ ACTION ITEMS")
        print("-" * 80)
        print(result["action_items"])

        print("\n🔑 KEY DECISIONS")
        print("-" * 80)
        print(result["key_decisions"])

        print("\n❓ OPEN QUESTIONS")
        print("-" * 80)
        print(result["open_questions"])

        print("\n⚠ RISKS & BLOCKERS")
        print("-" * 80)
        print(result["risks"])

        # ---------------------------------------------------------------------
        # Chat With Meeting
        # ---------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("💬 CHAT WITH MEETING")
        print("=" * 80)
        print("Type 'exit' to quit.\n")

        rag_chain = result["rag_chain"]

        while True:

            question = input("❓ You: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            answer = ask_question(
                rag_chain,
                question
            )

            print(f"\n🤖 Assistant: {answer}\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Process cancelled by user.")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print(str(e))