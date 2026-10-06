"""FAISS Retriever configured to fetch top K (default 3) relevant chunks."""

from typing import Optional
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS

from src.config.settings import get_settings


def get_faiss_retriever(
    vector_store: FAISS,
    top_k: Optional[int] = None,
    search_type: str = "similarity",
) -> VectorStoreRetriever:
    """Create a LangChain retriever from a FAISS vector store.

    Args:
        vector_store: FAISS vector store instance.
        top_k: Number of chunks to fetch (defaults to 3 as specified in architecture).
        search_type: Search type ('similarity', 'mmr', 'similarity_score_threshold').

    Returns:
        VectorStoreRetriever configured to fetch top K chunks.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k_chunks

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

