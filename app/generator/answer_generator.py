from typing import List
from langchain_core.documents import Document

from app.core.logger import logger
from app.generator.llm import get_llm
from app.generator.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class AnswerGenerator:
    """
    Generates grounded, context-aware answers using Ollama.
    Fixes:
    - Fiction vs factual clarity
    - Overview tone control
    - Reduced poetic drift
    """

    def __init__(self):
        self.llm = get_llm()

    def _build_context(self, documents: List[Document]) -> str:
        """
        Combine documents into a single context string with awareness header.
        """
        header = (
            "The following context comes from uploaded documents. "
            "Some documents may describe fictional stories or narratives, "
            "while others may contain factual information.\n\n"
        )

        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            index = doc.metadata.get("start_index", "n/a")

            parts.append(
                f"[{i}] Source: {source} (index: {index})\n"
                f"{doc.page_content}"
            )

        return header + "\n\n".join(parts)

    def _needs_structured_overview(self, question: str) -> bool:
        """
        Detect overview-style questions.
        """
        q = question.lower()
        return any(
            key in q
            for key in [
                "overview",
                "explain",
                "tell me about",
                "what is",
                "summary",
            ]
        )

    def generate(
        self,
        question: str,
        documents: List[Document],
    ) -> str:
        if not documents:
            return "I don't have enough information to answer this."

        context = self._build_context(documents)

        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + USER_PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )
        )

        # 🔒 Overview / explain guard
        if self._needs_structured_overview(question):
            prompt = (
                "Provide a clear, neutral overview first. "
                "Avoid poetic or dramatic language unless explicitly requested.\n\n"
                + prompt
            )

        logger.info("Generating answer using Ollama")

        response = self.llm.invoke(prompt)
        return response.strip()
