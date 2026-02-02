import os
from backend.pdf_parser import extract_pdf_text, PDFParseError


def load_story_file(filepath: str) -> str:
    """
    Load text content from supported story files.

    Supported:
    - .txt
    - .pdf

    Returns:
        str: Plain text content
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                return f.read()

    elif ext == ".pdf":
        return extract_pdf_text(filepath)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
