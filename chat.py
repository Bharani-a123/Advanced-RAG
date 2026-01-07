from app.loaders.folder_loader import FolderLoader
from app.chunking.splitter import DocumentChunker
from app.embeddings.embedder import ChromaEmbedder
from app.retrievers.hybrid_retriever import get_hybrid_retriever
from app.reranker.reranker import Reranker
from app.generator.answer_generator import AnswerGenerator


RAW_DOCS_DIR = "data/raw_docs"


def main():
    print("\n🧠 RAG CLI Chat")
    print(f"Using documents from: {RAW_DOCS_DIR}")
    print("Type 'exit' to quit\n")

    # 1️⃣ Load all docs from raw_docs
    loader = FolderLoader(RAW_DOCS_DIR)
    documents = loader.load()

    if not documents:
        print("❌ No documents found in data/raw_docs/")
        return

    # 2️⃣ Chunk
    chunker = DocumentChunker()
    chunks = chunker.split(documents)

    # 3️⃣ Embed (auto-persist)
    embedder = ChromaEmbedder()
    embedder.embed_and_store(chunks)

    # 4️⃣ Build retrieval + tools
    retriever = get_hybrid_retriever(chunks)
    reranker = Reranker(top_n=3)
    generator = AnswerGenerator()

    # 5️⃣ Chat loop
    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye")
            break

        retrieved_docs = retriever.get_relevant_documents(query)
        reranked_docs = reranker.rerank(query, retrieved_docs)
        answer = generator.generate(query, reranked_docs)

        print("\nAI:", answer, "\n")


if __name__ == "__main__":
    main()
