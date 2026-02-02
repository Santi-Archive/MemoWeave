from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError


class PDFParseError(Exception):
    pass


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract readable text from a PDF file.

    Args:
        pdf_path (str): Absolute path to PDF file

    Returns:
        str: Extracted plain text

    Raises:
        PDFParseError: If text extraction fails
    """
    try:
        text = extract_text(pdf_path)

        if not text or not text.strip():
            raise PDFParseError("PDF contains no extractable text (may be scanned).")

        return text.strip()

    except PDFSyntaxError as e:
        raise PDFParseError(f"Invalid PDF structure: {e}")

    except Exception as e:
        raise PDFParseError(f"Failed to parse PDF: {e}")
