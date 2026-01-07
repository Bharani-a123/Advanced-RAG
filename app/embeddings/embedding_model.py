from langchain_huggingface import HuggingFaceEmbeddings
from app.core.logger import logger


def get_embedding_model():
    """
    Returns a HuggingFace embedding model instance.
    """
    logger.info("Initializing HuggingFace embedding model: all-MiniLM-L6-v2")

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
