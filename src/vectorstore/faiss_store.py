"""FAISS Vector Store management, indexing, and persistence."""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from src.config.settings import get_settings
from src.embeddings.hf_embeddings import get_embedding_model


def create_vector_store(
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
    save_directory: Optional[Path | str] = None,
    index_name: str = "index",
) -> FAISS:
    """Create a new FAISS vector store from documents and save to disk.

    Args:
        documents: List of chunked Document objects to embed and index.
        embeddings: HuggingFace embeddings model instance.
        save_directory: Directory path to persist the FAISS index.
        index_name: Filename prefix for the FAISS index files.

    Returns:
        FAISS vector store instance.
    """
    settings = get_settings()
    embed_model = embeddings or get_embedding_model()
    target_dir = Path(save_directory or settings.vector_store_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    vector_store = FAISS.from_documents(documents=documents, embedding=embed_model)
    vector_store.save_local(str(target_dir), index_name=index_name)
    return vector_store


def load_vector_store(
    embeddings: Optional[Embeddings] = None,
    save_directory: Optional[Path | str] = None,
    index_name: str = "index",
) -> FAISS:
    """Load an existing FAISS vector store from disk.

    Args:
        embeddings: HuggingFace embeddings model instance.
        save_directory: Directory where the FAISS index is saved.
        index_name: Filename prefix for the FAISS index files.

    Returns:
        Loaded FAISS vector store instance.
    """
    settings = get_settings()
    embed_model = embeddings or get_embedding_model()
    target_dir = Path(save_directory or settings.vector_store_dir)

    index_file = target_dir / f"{index_name}.faiss"
    if not index_file.exists():
        raise FileNotFoundError(f"No FAISS index found at: {index_file}")

    return FAISS.load_local(
        str(target_dir),
        embeddings=embed_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def save_vector_store(
    vector_store: FAISS,
    save_directory: Optional[Path | str] = None,
    index_name: str = "index",
) -> None:
    """Save an in-memory FAISS vector store to disk."""
    settings = get_settings()
    target_dir = Path(save_directory or settings.vector_store_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(target_dir), index_name=index_name)


def get_document_vector_dir(doc_id: Optional[str] = None) -> Path:
    """Resolve vector store directory for a specific document ID with backward compatibility."""
    settings = get_settings()
    base_dir = settings.vector_store_dir

    if doc_id:
        clean_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in doc_id)
        sub_dir = base_dir / clean_id
        if (sub_dir / "index.faiss").exists():
            return sub_dir

    # Fallback to base vector store directory if index exists there
    if (base_dir / "index.faiss").exists():
        return base_dir

    if doc_id:
        clean_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in doc_id)
        return base_dir / clean_id

    return base_dir


def list_available_vector_stores() -> List[str]:
    """Scan and list all document IDs that have an indexed FAISS vector store."""
    settings = get_settings()
    base_dir = settings.vector_store_dir
    if not base_dir.exists():
        return []

    doc_ids = []
    # Check subdirectories
    for p in sorted(base_dir.iterdir()):
        if p.is_dir() and (p / "index.faiss").exists():
            doc_ids.append(p.name)

    # Check root vector directory fallback
    if (base_dir / "index.faiss").exists() and "Hands-On_Large_Language_Models" not in doc_ids:
        doc_ids.append("Hands-On_Large_Language_Models")

    return doc_ids


def get_or_create_vector_store(
    documents: Optional[List[Document]] = None,
    embeddings: Optional[Embeddings] = None,
    save_directory: Optional[Path | str] = None,
    index_name: str = "index",
) -> FAISS:
    """Load existing vector store if present on disk; otherwise create from documents."""
    settings = get_settings()
    target_dir = Path(save_directory or settings.vector_store_dir)
    index_file = target_dir / f"{index_name}.faiss"

    if index_file.exists():
        return load_vector_store(
            embeddings=embeddings,
            save_directory=target_dir,
            index_name=index_name,
        )

    if not documents:
        raise ValueError(
            f"No existing FAISS index at {index_file} and no documents provided to create one."
        )

    return create_vector_store(
        documents=documents,
        embeddings=embeddings,
        save_directory=target_dir,
        index_name=index_name,
    )

