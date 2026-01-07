from typing import List
from langchain_core.documents import Document

from app.loaders.base_loader import BaseLoader


class TextFileLoader(BaseLoader):
    """
    Loader for plain text (.txt) files.
    """

    def load(self) -> List[Document]:
        documents = []

        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()

        metadata = self._build_metadata(
            doc_type="text"
        )

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

        return documents
