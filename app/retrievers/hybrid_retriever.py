from typing import List
from collections import defaultdict

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import logger
from app.retrievers.vector_retriever import get_vector_retriever
from app.retrievers.bm25_retriever import get_bm25_retriever


class HybridRetriever:
    """
    Manual hybrid retriever:
    - Vector similarity search
    - BM25 keyword search
    - Weighted merge (LangChain v1.x compatible)
    """

    def __init__(self, chunks: List[Document]):
        self.vector_retriever = get_vector_retriever()
        self.bm25_retriever = get_bm25_retriever(chunks)

        self.vector_weight = settings.VECTOR_WEIGHT
        self.bm25_weight = settings.BM25_WEIGHT

        logger.info(
            "HybridRetriever initialized | "
            f"vector={self.vector_weight}, bm25={self.bm25_weight}"
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 🔑 LangChain v1.x uses invoke()
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        scores = defaultdict(float)
        doc_map = {}

        for doc in vector_docs:
            key = doc.page_content
            scores[key] += self.vector_weight
            doc_map[key] = doc

        for doc in bm25_docs:
            key = doc.page_content
            scores[key] += self.bm25_weight
            doc_map[key] = doc

        ranked_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = [doc_map[key] for key, _ in ranked_docs]

        logger.info(
            f"Hybrid retrieval completed | results={len(results)}"
        )

        return results


def get_hybrid_retriever(chunks: List[Document]) -> HybridRetriever:
    return HybridRetriever(chunks)
