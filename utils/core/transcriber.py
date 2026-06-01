import os
import logging
import tempfile
from typing import List

import whisper
import requests
from pydub import AudioSegment

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

SARVAM_PIECE_SECONDS = 25

_model = None

# -----------------------------------------------------------------------------
# Whisper Model Loader
# -----------------------------------------------------------------------------

def load_model():
    """
    Lazy-load Whisper model only once.
    """

    global _model

    if _model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        _model = whisper.load_model(WHISPER_MODEL)
        logger.info("Whisper model loaded successfully.")

    return _model


# -----------------------------------------------------------------------------
# Whisper Transcription
# -----------------------------------------------------------------------------

def transcribe_chunk_whisper(chunk_path: str) -> str:
    """
    Transcribe one WAV chunk using local Whisper.
    """

    try:
        model = load_model()

        result = model.transcribe(
            chunk_path,
            task="transcribe"
        )

        return result["text"]

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise


# -----------------------------------------------------------------------------
# Sarvam API Helper
# -----------------------------------------------------------------------------

def _send_to_sarvam(piece_path: str) -> str:
    """
    Send ≤30 sec WAV file to Sarvam.
    """

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as audio_file:

        files = {
            "file": (
                os.path.basename(piece_path),
                audio_file,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )

    if not response.ok:
        logger.error(
            f"Sarvam Error {response.status_code}: {response.text}"
        )
        response.raise_for_status()

    return response.json().get("transcript", "")


# -----------------------------------------------------------------------------
# Sarvam Transcription
# -----------------------------------------------------------------------------

def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Split audio into 25-second segments
    and send them to Sarvam API.
    """

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY not found in environment variables."
        )

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    transcript_parts = []

    total_pieces = (
        len(audio) + piece_ms - 1
    ) // piece_ms

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[start:start + piece_ms]

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            piece_path = temp_file.name

        try:
            piece.export(piece_path, format="wav")

            logger.info(
                f"Sarvam piece {i + 1}/{total_pieces}"
            )

            transcript_parts.append(
                _send_to_sarvam(piece_path)
            )

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(transcript_parts).strip()


# -----------------------------------------------------------------------------
# Engine Router
# -----------------------------------------------------------------------------

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:
    """
    Route transcription engine.

    english  -> Whisper
    hinglish -> Sarvam
    """

    language = language.lower()

    if language == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


# -----------------------------------------------------------------------------
# Batch Transcription
# -----------------------------------------------------------------------------

def transcribe_all(
    chunks: List[str],
    language: str = "english"
) -> str:
    """
    Transcribe all audio chunks and
    combine into one transcript.
    """

    engine = (
        "Sarvam AI"
        if language.lower() == "hinglish"
        else "Whisper"
    )

    logger.info(f"Using {engine}")

    transcript_parts = []

    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):

        logger.info(
            f"Transcribing chunk {idx}/{total}"
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        transcript_parts.append(text)

    logger.info("Transcription completed.")

    return " ".join(transcript_parts).strip()