from langchain_ollama import OllamaLLM
from app.core.config import settings
from app.core.logger import logger


def get_llm():
    """
    Returns an Ollama LLM instance.
    """
    logger.info(f"Initializing Ollama LLM: {settings.OLLAMA_MODEL}")

    return OllamaLLM(
        model=settings.OLLAMA_MODEL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )
