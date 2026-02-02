"""
Utility functions for the Temporal Memory Layer
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union
from backend.pdf_parser import extract_pdf_text, PDFParseError



def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file and return as dictionary.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Union[Dict[str, Any], List[Any]], filepath: str) -> None:
    """
    Save dictionary or list to JSON file.
    
    Args:
        data: Dictionary or list to save
        filepath: Path where to save JSON file
    """
    dir_path = os.path.dirname(filepath)
    if dir_path:
        ensure_directory(dir_path)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_reference_date() -> str:
    """
    Get current date as reference for temporal normalization.
    
    Returns:
        Current date in YYYY-MM-DD format
    """
    return datetime.now().strftime("%Y-%m-%d")


def read_text_file(filepath: str) -> str:
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
        raise ValueError(f"Unsupported input file type: {ext}")

