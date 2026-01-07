from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import logger
from app.chunking.strategies import ChunkingStrategy


class DocumentChunker:
    """
    Splits documents into chunks while preserving metadata.
    Hybrid-search friendly.
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        if self.strategy == ChunkingStrategy.RECURSIVE:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True,  # useful for reranking & citations
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        logger.info(
            f"Chunker initialized | size={self.chunk_size} overlap={self.chunk_overlap}"
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        Metadata is preserved automatically by LangChain.
        """
        if not documents:
            logger.warning("No documents received for chunking")
            return []

        chunks = self.splitter.split_documents(documents)

        logger.info(
            f"Chunking completed | input_docs={len(documents)} chunks={len(chunks)}"
        )

        return chunks
