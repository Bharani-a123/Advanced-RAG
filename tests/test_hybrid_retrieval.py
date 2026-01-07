from app.loaders.text_loader import TextFileLoader
from app.chunking.splitter import DocumentChunker
from app.embeddings.embedder import ChromaEmbedder
from app.retrievers.hybrid_retriever import get_hybrid_retriever


def test_hybrid_retrieval():
    # Load & chunk
    loader = TextFileLoader("data/raw_docs/sample.txt")
    docs = loader.load()

    chunker = DocumentChunker()
    chunks = chunker.split(docs)

    # Ensure vectors exist
    embedder = ChromaEmbedder()
    embedder.embed_and_store(chunks)

    # Hybrid retriever
    retriever = get_hybrid_retriever(chunks)

    # Test query
    query = "Artificial Intelligence in 2026"
    results = retriever.get_relevant_documents(query)

    print(f"\nRetrieved {len(results)} documents\n")

    for i, doc in enumerate(results[:5], 1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:150])
        print("Metadata:", doc.metadata)


if __name__ == "__main__":
    test_hybrid_retrieval()
