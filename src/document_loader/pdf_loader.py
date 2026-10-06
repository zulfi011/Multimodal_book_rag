"""PDF document loader using PDFPlumberLoader for high fidelity text extraction."""

from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_pdf_document(file_path: str | Path) -> List[Document]:
    """Load and extract text from a PDF file using PDFPlumberLoader.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        List of LangChain Document objects with extracted page content and metadata.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file not found at path: {path_obj}")

    import logging
    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    # Direct import to keep module import times fast
    from langchain_community.document_loaders.pdf import PDFPlumberLoader

    loader = PDFPlumberLoader(str(path_obj))
    documents = loader.load()
    return documents

