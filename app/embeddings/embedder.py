from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma


from app.core.config import settings
from app.core.logger import logger
from app.embeddings.embedding_model import get_embedding_model


class ChromaEmbedder:
    """
    Handles embedding documents and storing them in ChromaDB.
    Compatible with Chroma 0.4+ (auto-persistence).
    """

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.persist_dir = settings.CHROMA_PERSIST_DIR

        logger.info(
            f"Using ChromaDB persist directory: {self.persist_dir}"
        )

    def embed_and_store(self, documents: List[Document]) -> Chroma:
        """
        Embed documents and store them in ChromaDB.
        Auto-persistence is handled internally by Chroma.
        """
        if not documents:
            raise ValueError("No documents provided for embedding")

        logger.info(f"Embedding {len(documents)} chunks into ChromaDB")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir,
        )

        logger.info("ChromaDB auto-persistence completed")
        return vectorstore

    def load_existing(self) -> Chroma:
        """
        Load an existing ChromaDB vector store.
        """
        logger.info("Loading existing ChromaDB vector store")

        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_model,
        )
