"""PDF text extraction. No LLM logic."""

from pathlib import Path
from pypdf import PdfReader


def extract_text(pdf_path: str | Path) -> str:
    """Return the concatenated text of all pages in a PDF.

    Raises FileNotFoundError if the path doesn't exist.
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    reader = PdfReader(str(p))
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts).strip()
