import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # ===============================
    # LLM (Ollama)
    # ===============================
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", 0.2))

    # ===============================
    # MongoDB Atlas (Cloud)
    # ===============================
    MONGODB_URI: str = os.getenv("MONGODB_URI")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "advanced_rag")

    # ===============================
    # Vector Store (Chroma)
    # ===============================
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", "data/chroma_db"
    )

    # ===============================
    # Hybrid Search Settings
    # ===============================
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", 8))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", 8))

    # Hybrid weights (VERY IMPORTANT)
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", 0.6))
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", 0.4))

    # ===============================
    # Chunking Defaults
    # ===============================
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

settings = Settings()
