import os
from typing import List
from langchain_core.documents import Document

from app.loaders.text_loader import TextFileLoader
from app.loaders.pdf_loader import PDFFileLoader
from app.core.logger import logger


class FolderLoader:
    """
    Loads all supported documents from data/raw_docs.
    """

    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load(self) -> List[Document]:
        if not os.path.isdir(self.folder_path):
            raise ValueError(f"Folder not found: {self.folder_path}")

        documents: List[Document] = []

        for file in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file)

            if file.lower().endswith(".pdf"):
                logger.info(f"Loading PDF: {file}")
                loader = PDFFileLoader(file_path)
                documents.extend(loader.load())

            elif file.lower().endswith(".txt"):
                logger.info(f"Loading TXT: {file}")
                loader = TextFileLoader(file_path)
                documents.extend(loader.load())

        logger.info(f"Loaded {len(documents)} documents from folder")
        return documents
