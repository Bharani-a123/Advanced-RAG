from sentence_transformers import CrossEncoder
from app.core.logger import logger


def get_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
):
    """
    Load a cross-encoder model for reranking.
    """
    logger.info(f"Loading cross-encoder model: {model_name}")

    model = CrossEncoder(model_name)
    return model
