"""Document loading module."""

from .pdf_loader import load_pdf_document
from .image_extractor import extract_page_figures, get_figures_for_pages

__all__ = ["load_pdf_document", "extract_page_figures", "get_figures_for_pages"]

