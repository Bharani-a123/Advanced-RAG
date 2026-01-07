from app.loaders.text_loader import TextFileLoader
from app.chunking.splitter import DocumentChunker


def test_chunking():
    loader = TextFileLoader("data/raw_docs/sample.txt")
    docs = loader.load()

    chunker = DocumentChunker()
    chunks = chunker.split(docs)

    print("Original docs:", len(docs))
    print("Chunks:", len(chunks))

    print("\nSample chunk:")
    print(chunks[0].page_content[:200])
    print("Metadata:", chunks[0].metadata)


if __name__ == "__main__":
    test_chunking()
