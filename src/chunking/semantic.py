"""Semantic chunking using LangChain SemanticChunker and HuggingFace Embeddings."""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config.settings import get_settings
from src.embeddings.hf_embeddings import get_embedding_model


def get_semantic_chunker(
    embeddings: Optional[Embeddings] = None,
    breakpoint_threshold_type: Optional[str] = None,
    breakpoint_threshold_amount: Optional[float] = None,
):
    """Instantiate a SemanticChunker configured with HuggingFace embeddings.

    Args:
        embeddings: Embeddings model instance. If None, initialized from settings.
        breakpoint_threshold_type: Threshold type ('percentile', 'standard_deviation', 'interquartile').
        breakpoint_threshold_amount: Threshold amount corresponding to threshold type.

    Returns:
        SemanticChunker instance.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    settings = get_settings()
    embed_model = embeddings or get_embedding_model()
    threshold_type = breakpoint_threshold_type or settings.breakpoint_threshold_type
    threshold_amount = (
        breakpoint_threshold_amount
        if breakpoint_threshold_amount is not None
        else settings.breakpoint_threshold_amount
    )

    return SemanticChunker(
        embeddings=embed_model,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold_amount,
    )


def split_documents_semantically(
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
    show_progress: bool = True,
) -> List[Document]:
    """Split a list of documents into semantic chunks.

    Args:
        documents: Raw documents loaded from PDF.
        embeddings: Optional embeddings model to determine semantic breakpoints.
        show_progress: Whether to display a tqdm progress bar during chunking.

    Returns:
        List of semantically chunked Document objects.
    """
    chunker = get_semantic_chunker(embeddings=embeddings)

    if not show_progress:
        return chunker.split_documents(documents)

    try:
        from tqdm import tqdm

        all_chunks: List[Document] = []
        for doc in tqdm(documents, desc="Semantic Chunking", unit="page"):
            if doc.page_content and doc.page_content.strip():
                try:
                    page_chunks = chunker.split_documents([doc])
                    all_chunks.extend(page_chunks)
                except Exception:
                    # Fallback to keep whole doc if edge case in empty/special characters
                    all_chunks.append(doc)
        return all_chunks
    except ImportError:
        return chunker.split_documents(documents)

