from typing import List
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader
from app.loaders.base_loader import BaseLoader


class PDFFileLoader(BaseLoader):
    """
    Loader for PDF files.
    """

    def load(self) -> List[Document]:
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()

        documents: List[Document] = []

        for page in pages:
            metadata = self._build_metadata(
                doc_type="pdf",
                page=page.metadata.get("page", None)
            )

            documents.append(
                Document(
                    page_content=page.page_content,
                    metadata=metadata
                )
            )

        return documents
