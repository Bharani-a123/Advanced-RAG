from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document



class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load file and return LangChain Document objects.
        """
        pass

    def _build_metadata(self, **kwargs) -> dict:
        """
        Common metadata builder.
        """
        return {
            "source": self.file_path,
            **kwargs
        }
