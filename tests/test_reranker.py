from app.loaders.text_loader import TextFileLoader
from app.chunking.splitter import DocumentChunker
from app.embeddings.embedder import ChromaEmbedder
from app.retrievers.hybrid_retriever import get_hybrid_retriever
from app.reranker.reranker import Reranker


def test_reranker():
    # Load & chunk
    loader = TextFileLoader("data/raw_docs/sample.txt")
    docs = loader.load()

    chunker = DocumentChunker()
    chunks = chunker.split(docs)

    # Ensure vectors exist
    embedder = ChromaEmbedder()
    embedder.embed_and_store(chunks)

    # Hybrid retrieval
    retriever = get_hybrid_retriever(chunks)
    query = "Artificial Intelligence in 2026"
    retrieved_docs = retriever.get_relevant_documents(query)

    print(f"\nRetrieved before rerank: {len(retrieved_docs)}")

    # Rerank
    reranker = Reranker(top_n=3)
    reranked_docs = reranker.rerank(query, retrieved_docs)

    print(f"\nAfter rerank: {len(reranked_docs)}\n")

    for i, doc in enumerate(reranked_docs, 1):
        print(f"--- Reranked Result {i} ---")
        print(doc.page_content[:200])
        print("Metadata:", doc.metadata)


if __name__ == "__main__":
    test_reranker()
