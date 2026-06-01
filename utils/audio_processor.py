import os
import logging
from pathlib import Path

import yt_dlp
from pydub import AudioSegment
from pydub.utils import which

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# FFmpeg Check
# -----------------------------------------------------------------------------

if which("ffmpeg") is None:
    raise EnvironmentError(
        "FFmpeg not found. Please install FFmpeg and add it to PATH."
    )

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# YouTube Audio Download
# -----------------------------------------------------------------------------

def download_youtube_audio(url: str) -> str:
    """
    Download YouTube audio and convert it to WAV format.

    Args:
        url (str): YouTube URL

    Returns:
        str: Path to downloaded WAV file
    """

    logger.info("Downloading audio from YouTube...")

    output_template = str(DOWNLOAD_DIR / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            filename = (
                ydl.prepare_filename(info)
                .replace(".webm", ".wav")
                .replace(".m4a", ".wav")
                .replace(".mp4", ".wav")
            )

        logger.info("Download completed.")
        return filename

    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        raise


# -----------------------------------------------------------------------------
# Convert Local File to WAV
# -----------------------------------------------------------------------------

def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to 16kHz mono WAV.

    Args:
        input_path (str): Input file path

    Returns:
        str: Converted WAV file path
    """

    logger.info("Converting file to WAV format...")

    try:
        output_path = str(
            Path(input_path).with_suffix("")
        ) + "_converted.wav"

        audio = AudioSegment.from_file(input_path)

        audio = (
            audio
            .set_channels(1)
            .set_frame_rate(16000)
        )

        audio.export(output_path, format="wav")

        logger.info("Conversion completed.")
        return output_path

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


# -----------------------------------------------------------------------------
# Audio Chunking
# -----------------------------------------------------------------------------

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list[str]:
    """
    Split large audio into smaller chunks.

    Args:
        wav_path (str): WAV file path
        chunk_minutes (int): Chunk duration in minutes

    Returns:
        list[str]: Chunk file paths
    """

    logger.info("Creating audio chunks...")

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    base_name = os.path.splitext(wav_path)[0]

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{base_name}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    logger.info(f"{len(chunks)} chunk(s) created.")

    return chunks


# -----------------------------------------------------------------------------
# Main Processing Function
# -----------------------------------------------------------------------------

def process_input(source: str) -> list[str]:
    """
    Process either:
    - YouTube URL
    - Local audio/video file

    Returns:
        list[str]: List of chunk paths
    """

    source = source.strip()

    if "youtube.com" in source or "youtu.be" in source:
        logger.info("Detected YouTube URL.")
        wav_path = download_youtube_audio(source)

    else:
        logger.info("Detected local file.")
        wav_path = convert_to_wav(source)

    chunks = chunk_audio(wav_path)

    logger.info("Audio processing completed successfully.")

    return chunks