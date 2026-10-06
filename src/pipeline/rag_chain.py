"""End-to-end Conversational RAG chain assembling FAISS Retriever, diagram extraction, and DeepSeek R1."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from src.retriever.faiss_retriever import get_faiss_retriever
from src.llm.deepseek import get_deepseek_llm
from src.document_loader.image_extractor import extract_page_figures


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    query: str
    answer: str
    source_documents: List[Document]
    images: List[str] = field(default_factory=list)
    standalone_query: Optional[str] = None


CONTEXTUALIZE_Q_TEMPLATE = """Given the recent conversation history and the latest user question which might reference context in the history, formulate a standalone search question that can be understood without the chat history. Do NOT answer the question, just return the reformulated question if needed, otherwise return it as is.

Recent Conversation:
{chat_history}

Latest Question: {question}

Standalone Question:"""

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_template(CONTEXTUALIZE_Q_TEMPLATE)

CONVERSATIONAL_RAG_TEMPLATE = """You are an expert AI assistant specialized in analyzing and explaining technical books.
Answer the user's question accurately, thoroughly, and comprehensively using ONLY the context excerpts provided below.
If technical diagrams or figures are referenced in the context, refer to them explicitly (e.g. "As illustrated in Figure on Page X").
Maintain a natural, coherent conversational dialogue by taking into account the recent conversation history when relevant.
If the answer cannot be found in the context, truthfully state that you do not know based on the provided material.

Context (Top relevant excerpts from the book):
{context}

Recent Conversation History:
{chat_history}

Question:
{question}

Answer:"""

CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_template(CONVERSATIONAL_RAG_TEMPLATE)


def format_docs(docs: List[Document]) -> str:
    """Format a list of documents into a single context string."""
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", doc.metadata.get("page_number", "N/A"))
        formatted_chunks.append(f"[Chunk {i} | Page {page}]:\n{doc.page_content.strip()}")
    return "\n\n".join(formatted_chunks)


def format_chat_history(messages: Optional[List[Dict[str, Any]]], max_turns: int = 4) -> str:
    """Format the last N conversation turns into a compact text block."""
    if not messages:
        return "None"

    # Take the last 2 * max_turns messages (user + assistant pairs)
    recent = messages[-(max_turns * 2):]
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = str(m.get("content", "")).strip()
        # Compact lengthy prior assistant messages to conserve token budget
        if role == "Assistant" and len(content) > 350:
            content = content[:350] + "... [previous summary truncated]"
        lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "None"


class BookRAGPipeline:
    """End-to-end pipeline coordinating query contextualization, retrieval, diagram extraction, and generation."""

    def __init__(
        self,
        vector_store: FAISS,
        top_k: int = 3,
        llm_provider: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.retriever = get_faiss_retriever(vector_store, top_k=top_k)
        self.llm = get_deepseek_llm(provider=llm_provider)

        # Contextualization chain: rewrites follow-up questions into standalone queries
        self._contextualize_chain = (
            CONTEXTUALIZE_Q_PROMPT
            | self.llm
            | StrOutputParser()
        )

        # Final answer synthesis chain
        self._qa_chain = (
            CONVERSATIONAL_RAG_PROMPT
            | self.llm
        )

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> RAGResponse:
        """Query the pipeline with conversational memory support.

        Args:
            question: Latest user question string.
            chat_history: Optional list of previous chat messages.

        Returns:
            RAGResponse containing query, generated answer, sources, images, and standalone query.
        """
        history_text = format_chat_history(chat_history)
        standalone_q = question

        # If previous chat history exists, reformulate follow-up into a standalone question
        if chat_history and len(chat_history) > 0 and history_text != "None":
            try:
                rewritten = self._contextualize_chain.invoke({
                    "chat_history": history_text,
                    "question": question,
                })
                if rewritten and rewritten.strip():
                    standalone_q = rewritten.strip().replace("\n", " ")
            except Exception:
                standalone_q = question

        # Fetch Top K Chunks using the standalone query
        docs = self.retriever.invoke(standalone_q)

        # Extract and link any figures on the pages of the retrieved chunks
        associated_images: List[str] = []
        for d in docs:
            pdf_path = d.metadata.get("source") or d.metadata.get("file_path")
            page_num = d.metadata.get("page")
            if pdf_path and page_num is not None:
                figs = extract_page_figures(pdf_path=pdf_path, page_num=int(page_num))
                for f in figs:
                    if f not in associated_images:
                        associated_images.append(f)

        # Generate answer from DeepSeek R1 with full conversational context
        raw_output = self._qa_chain.invoke({
            "context": format_docs(docs),
            "chat_history": history_text,
            "question": question,
        })

        # Ensure answer is strictly extracted and never blank
        answer = ""
        if isinstance(raw_output, str):
            answer = raw_output.strip()
        else:
            content = getattr(raw_output, "content", "")
            if content and str(content).strip():
                answer = str(content).strip()
            else:
                kwargs = getattr(raw_output, "additional_kwargs", {}) or {}
                for key in ["reasoning", "reasoning_content", "thought"]:
                    val = kwargs.get(key)
                    if val and str(val).strip():
                        answer = str(val).strip()
                        break

        if not answer:
            answer = "Based on the provided excerpts, no answer could be formulated."

        return RAGResponse(
            query=question,
            answer=answer,
            source_documents=docs,
            images=associated_images,
            standalone_query=standalone_q if standalone_q != question else None,
        )


def create_rag_chain(
    vector_store: FAISS,
    top_k: int = 3,
    llm_provider: Optional[str] = None,
) -> BookRAGPipeline:
    """Factory function to build a BookRAGPipeline instance."""
    return BookRAGPipeline(
        vector_store=vector_store,
        top_k=top_k,
        llm_provider=llm_provider,
    )
