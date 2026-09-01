from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path


def _extract_pdf_with_pypdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _extract_pdf_with_pymupdf(content: bytes) -> str:
    try:
        import fitz  # PyMuPDF  # type: ignore[import-not-found]

        doc = fitz.open(stream=content, filetype="pdf")
        try:
            return "\n".join((page.get_text() or "") for page in doc)
        finally:
            doc.close()
    except Exception:
        return ""


def _extract_pdf_with_ocr(content: bytes) -> str:
    if not shutil.which("tesseract"):
        return ""
    try:
        import pytesseract  # type: ignore[import-not-found]
        from pdf2image import convert_from_bytes  # type: ignore[import-not-found]

        images = convert_from_bytes(content, dpi=200, first_page=1, last_page=15)
        chunks: list[str] = []
        for img in images:
            chunks.append(pytesseract.image_to_string(img) or "")
        return "\n".join(chunks)
    except Exception:
        return ""


def _best_pdf_text(content: bytes) -> str:
    a = _extract_pdf_with_pypdf(content).strip()
    b = _extract_pdf_with_pymupdf(content).strip()
    best = a if len(a) >= len(b) else b
    # Very short text is usually useless (or missing glyphs); try OCR for scanned PDFs.
    if len(best) >= 40:
        return best
    ocr = _extract_pdf_with_ocr(content).strip()
    return ocr if len(ocr) > len(best) else best


def extract_text_from_upload(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        return _best_pdf_text(content)

    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore[import-not-found]

            document = Document(BytesIO(content))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception:
            return ""

    return ""
