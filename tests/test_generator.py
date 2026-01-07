from app.loaders.text_loader import TextFileLoader
from app.chunking.splitter import DocumentChunker
from app.embeddings.embedder import ChromaEmbedder
from app.retrievers.hybrid_retriever import get_hybrid_retriever
from app.reranker.reranker import Reranker
from app.generator.answer_generator import AnswerGenerator


def test_generator():
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
    query = "What changed in Artificial Intelligence by 2026?"
    retrieved_docs = retriever.get_relevant_documents(query)

    # Rerank
    reranker = Reranker(top_n=3)
    reranked_docs = reranker.rerank(query, retrieved_docs)

    # Generate answer
    generator = AnswerGenerator()
    answer = generator.generate(query, reranked_docs)

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    test_generator()
