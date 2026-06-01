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
# Main Pipeline
# -----------------------------------------------------------------------------

def main():
    try:
        print("\n" + "=" * 80)
        print("🚀 AI MEETING ASSISTANT")
        print("=" * 80)

        # ---------------------------------------------------------------------
        # User Input
        # ---------------------------------------------------------------------

        source = input(
            "\nEnter YouTube URL or Local File Path:\n> "
        ).strip()

        if not source:
            raise ValueError("No input provided.")

        language = input(
            "\nChoose language [english / hinglish] (default: english):\n> "
        ).strip().lower()

        if not language:
            language = "english"

        if language not in ["english", "hinglish"]:
            print("⚠ Invalid language. Using English.")
            language = "english"

        # ---------------------------------------------------------------------
        # Audio Processing
        # ---------------------------------------------------------------------

        print("\n🎧 Processing Audio...")

        chunks = process_input(source)

        print(f"✅ Created {len(chunks)} chunk(s)")

        # ---------------------------------------------------------------------
        # Transcription
        # ---------------------------------------------------------------------

        print("\n📝 Generating Transcript...")

        transcript = transcribe_all(
            chunks,
            language=language
        )

        print("\n" + "=" * 80)
        print("📝 TRANSCRIPT PREVIEW")
        print("=" * 80)

        preview = (
            transcript[:1000] + "..."
            if len(transcript) > 1000
            else transcript
        )

        print(preview)

        # ---------------------------------------------------------------------
        # Summary Generation
        # ---------------------------------------------------------------------

        print("\n🤖 Generating Summary...")

        title = generate_title(transcript)
        summary = summarize(transcript)

        print("\n" + "=" * 80)
        print(f"📌 TITLE: {title}")
        print("=" * 80)

        print("\n📋 SUMMARY")
        print("-" * 80)
        print(summary)

        # ---------------------------------------------------------------------
        # Information Extraction
        # ---------------------------------------------------------------------

        print("\n🔍 Extracting Insights...")

        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        risks = extract_risks(transcript)

        # ---------------------------------------------------------------------
        # Action Items
        # ---------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("✅ ACTION ITEMS")
        print("=" * 80)
        print(action_items)

        # ---------------------------------------------------------------------
        # Key Decisions
        # ---------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("🔑 KEY DECISIONS")
        print("=" * 80)
        print(decisions)

        # ---------------------------------------------------------------------
        # Open Questions
        # ---------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("❓ OPEN QUESTIONS")
        print("=" * 80)
        print(questions)

        # ---------------------------------------------------------------------
        # Risks & Blockers
        # ---------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("⚠ RISKS & BLOCKERS")
        print("=" * 80)
        print(risks)

        # ---------------------------------------------------------------------
        # Build RAG Knowledge Base
        # ---------------------------------------------------------------------

        print("\n📚 Building Meeting Knowledge Base...")

        rag_chain = build_rag_chain(transcript)

        # ---------------------------------------------------------------------
        # Chat With Meeting
        # ---------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("💬 CHAT WITH MEETING")
        print("=" * 80)
        print("Type 'exit' to quit.\n")

        while True:

            question = input(
                "❓ Ask Question: "
            ).strip()

            if question.lower() == "exit":
                break

            if not question:
                continue

            answer = ask_question(
                rag_chain,
                question
            )

            print(f"\n🤖 {answer}\n")

        print("\n🎉 Processing Completed Successfully!")

    except KeyboardInterrupt:

        print(
            "\n\n⚠ Process cancelled by user."
        )

    except Exception as e:

        print("\n" + "=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print(str(e))


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()