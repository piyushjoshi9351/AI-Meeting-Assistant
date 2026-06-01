import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Directory Setup
# -----------------------------------------------------------------------------

Path(CHROMA_DIR).mkdir(
    parents=True,
    exist_ok=True
)

# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load embedding model.
    """

    logger.info(
        f"Loading embedding model: {EMBEDDING_MODEL}"
    )

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# -----------------------------------------------------------------------------
# Build Vector Store
# -----------------------------------------------------------------------------

def build_vector_store(
    transcript: str
) -> Chroma:
    """
    Create Chroma vector database
    from meeting transcript.
    """

    logger.info(
        "Building vector store..."
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_text(
        transcript
    )

    logger.info(
        f"Created {len(chunks)} chunks"
    )

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_index": idx
            }
        )
        for idx, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    logger.info(
        "Vector store created successfully."
    )

    return vector_store

# -----------------------------------------------------------------------------
# Load Existing Vector Store
# -----------------------------------------------------------------------------

def load_vector_store() -> Chroma:
    """
    Load persisted Chroma database.
    """

    logger.info(
        "Loading vector store..."
    )

    embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

# -----------------------------------------------------------------------------
# Retriever
# -----------------------------------------------------------------------------

def get_retriever(
    vector_store: Chroma,
    k: int = 4
):
    """
    Create retriever for RAG.
    """

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        }
    )

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def vector_store_exists() -> bool:
    """
    Check if vector database exists.
    """

    return Path(CHROMA_DIR).exists()