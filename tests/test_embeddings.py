from app.loaders.text_loader import TextFileLoader
from app.chunking.splitter import DocumentChunker
from app.embeddings.embedder import ChromaEmbedder


def test_embeddings():
    # Load document
    loader = TextFileLoader("data/raw_docs/sample.txt")
    docs = loader.load()

    # Chunk document
    chunker = DocumentChunker()
    chunks = chunker.split(docs)

    # Embed and store
    embedder = ChromaEmbedder()
    vectorstore = embedder.embed_and_store(chunks)

    print("Embedding test completed.")
    print("Number of vectors:", vectorstore._collection.count())


if __name__ == "__main__":
    test_embeddings()
