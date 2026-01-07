from enum import Enum


class ChunkingStrategy(str, Enum):
    """
    Supported chunking strategies.
    """
    RECURSIVE = "recursive"
