from typing import List, Tuple
from langchain_core.documents import Document

from app.core.logger import logger
from app.reranker.cross_encoder import get_cross_encoder


class Reranker:
    """
    Reranks retrieved documents using a cross-encoder.
    """

    def __init__(self, top_n: int = 5):
        self.model = get_cross_encoder()
        self.top_n = top_n

        logger.info(f"Reranker initialized | top_n={self.top_n}")

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        if not documents:
            return []

        # Prepare query-doc pairs
        pairs: List[Tuple[str, str]] = [
            (query, doc.page_content) for doc in documents
        ]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to documents
        scored_docs = list(zip(documents, scores))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Keep top-N
        top_docs = [doc for doc, _ in scored_docs[: self.top_n]]

        logger.info(
            f"Reranking completed | input={len(documents)} output={len(top_docs)}"
        )

        return top_docs
