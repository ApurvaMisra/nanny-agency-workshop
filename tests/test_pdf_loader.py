from pathlib import Path
from reportlab.pdfgen import canvas
from nanny_workshop.pdf_loader import extract_text


def make_sample_pdf(path: Path, text: str) -> None:
    c = canvas.Canvas(str(path))
    for i, line in enumerate(text.splitlines()):
        c.drawString(72, 720 - i * 14, line)
    c.save()


def test_extract_text_returns_pdf_content(tmp_path):
    pdf = tmp_path / "sample.pdf"
    make_sample_pdf(pdf, "Hello Nanny\nLine two")
    text = extract_text(pdf)
    assert "Hello Nanny" in text
    assert "Line two" in text


def test_extract_text_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "nope.pdf")
