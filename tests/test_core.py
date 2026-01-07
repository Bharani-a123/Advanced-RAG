from app.core.config import settings
from app.core.logger import logger

def test_core():
    logger.info("Testing core configuration...")
    print("Ollama model:", settings.OLLAMA_MODEL)
    print("Hybrid weights:", settings.VECTOR_WEIGHT, settings.BM25_WEIGHT)

test_core()
