from langchain_chroma import Chroma
from app.core.config import settings
from app.embeddings.embedding_model import get_embedding_model
from app.core.logger import logger


def get_vector_retriever():
    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.VECTOR_TOP_K}
    )

    logger.info(
        f"Vector retriever ready | top_k={settings.VECTOR_TOP_K}"
    )

    return retriever
