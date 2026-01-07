from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from app.core.config import settings
from app.core.logger import logger


def get_bm25_retriever(chunks: List[Document]):
    """
    In-memory BM25 retriever (rebuilt per session).
    """
    if not chunks:
        raise ValueError("BM25 retriever requires chunks")

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = settings.BM25_TOP_K

    logger.info(
        f"BM25 retriever ready | top_k={settings.BM25_TOP_K}"
    )

    return retriever
