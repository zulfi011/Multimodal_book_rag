"""HuggingFace Embeddings model wrapper for semantic representations."""

from typing import Optional
from langchain_core.embeddings import Embeddings
from src.config.settings import get_settings


def get_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Embeddings:
    """Initialize and return HuggingFace embeddings model.

    Args:
        model_name: Name or path of the HuggingFace model (defaults to settings).
        device: Device to run embeddings on ('cpu', 'cuda', 'mps').

    Returns:
        HuggingFaceEmbeddings instance.
    """
    settings = get_settings()
    target_model = model_name or settings.embedding_model_name
    target_device = device or settings.embedding_device

    # First attempt loading from local cache to avoid network roundtrips
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        model_kwargs = {"device": target_device}
        # Try local cache first
        try:
            return HuggingFaceEmbeddings(
                model_name=target_model,
                model_kwargs={"device": target_device, "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception:
            return HuggingFaceEmbeddings(
                model_name=target_model,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=target_model,
            model_kwargs={"device": target_device},
            encode_kwargs={"normalize_embeddings": True},
        )

