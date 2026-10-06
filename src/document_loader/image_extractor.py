"""Extract and crop high-resolution diagrams and figures from PDF pages."""

import logging
from pathlib import Path
from typing import List, Dict

# Suppress verbose pdfminer logs
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def extract_page_figures(
    pdf_path: str | Path,
    page_num: int,
    output_dir: str | Path = "data/images",
    min_width: int = 100,
    min_height: int = 80,
    scale: float = 2.0,
) -> List[str]:
    """Extract and crop all significant figures/diagrams from a specific PDF page.

    Args:
        pdf_path: Path to the target PDF file.
        page_num: 0-indexed page number (from Document.metadata['page']).
        output_dir: Directory to save cropped figure images.
        min_width: Minimum width in points to ignore small icons/bullets.
        min_height: Minimum height in points to ignore small icons/lines.
        scale: Resolution scale factor for rendering (2.0 gives crisp high-res output).

    Returns:
        List of file paths to the extracted figure images on disk.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        return []

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Check for already cached images for this page
    cached_images = sorted(list(out_path.glob(f"page_{page_num}_fig_*.png")))
    if cached_images:
        return [str(p) for p in cached_images]

    extracted_image_paths = []

    try:
        import pdfplumber
        import pypdfium2 as pdfium

        with pdfplumber.open(str(pdf_path_obj)) as pdf:
            if page_num < 0 or page_num >= len(pdf.pages):
                return []
            page = pdf.pages[page_num]

            # Filter for significant figures (exclude tiny icons, page rules, bullets)
            valid_images = [
                img
                for img in page.images
                if (img.get("width", 0) >= min_width and img.get("height", 0) >= min_height)
            ]

            if not valid_images:
                return []

            # Render page at high resolution using pypdfium2
            pdf_doc = pdfium.PdfDocument(str(pdf_path_obj))
            rendered_page = pdf_doc[page_num]
            pil_image = rendered_page.render(scale=scale).to_pil()

            for idx, img_info in enumerate(valid_images, start=1):
                # Calculate bounding box coordinates scaled by render factor
                x0 = max(0, int(img_info["x0"] * scale))
                top = max(0, int(img_info["top"] * scale))
                x1 = min(pil_image.width, int(img_info["x1"] * scale))
                bottom = min(pil_image.height, int(img_info["bottom"] * scale))

                # Ensure valid crop area
                if x1 > x0 and bottom > top:
                    cropped = pil_image.crop((x0, top, x1, bottom))
                    save_file = out_path / f"page_{page_num}_fig_{idx}.png"
                    cropped.save(str(save_file), format="PNG")
                    extracted_image_paths.append(str(save_file))

    except Exception as e:
        # Graceful fallback: return whatever was extracted without failing RAG
        logging.getLogger(__name__).warning(
            f"Failed to extract figures on page {page_num}: {e}"
        )

    return extracted_image_paths


def get_figures_for_pages(
    pdf_path: str | Path,
    page_numbers: List[int],
    output_dir: str | Path = "data/images",
) -> Dict[int, List[str]]:
    """Extract figures for multiple unique pages."""
    unique_pages = sorted(list(set(page_numbers)))
    results: Dict[int, List[str]] = {}
    for p in unique_pages:
        figures = extract_page_figures(pdf_path=pdf_path, page_num=p, output_dir=output_dir)
        if figures:
            results[p] = figures
    return results

