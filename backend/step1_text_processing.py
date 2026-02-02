"""
Step 1: Text Processing
Prepares raw story text for deeper analysis through chapter segmentation,
sentence tokenization, word tokenization, and linguistic annotation.
"""

import re
import spacy
from typing import List, Dict, Any
from .utils import save_json, ensure_directory


def segment_chapters(text: str) -> List[Dict[str, Any]]:
    """
    Split text using chapter markers like "Chapter 1", "CHAPTER 2", etc.
    
    Args:
        text: Raw story text
        
    Returns:
        List of chapter dictionaries with unique identifiers
    """
    chapters = []
    
    # Pattern to match various chapter formats
    # Matches: "Chapter 1", "CHAPTER 2", "Chapter One", "Chapter I", etc.
    chapter_pattern = r'(?i)^(?:Chapter|CHAP\.?)\s+([A-Z0-9IVX]+|[A-Za-z]+)'
    
    # Split text by chapter markers
    parts = re.split(chapter_pattern, text, flags=re.MULTILINE)
    
    # If no chapters found, treat entire text as one chapter
    if len(parts) == 1:
        chapters.append({
            "chapter_id": "chapter_1",
            "chapter_number": "1",
            "title": "Chapter 1",
            "text": text.strip()
        })
        return chapters
    
    # Process split parts (alternating between chapter markers and content)
    chapter_num = 1
    for i in range(1, len(parts), 2):
        if i < len(parts) - 1:
            chapter_marker = parts[i]
            chapter_text = parts[i + 1] if i + 1 < len(parts) else ""
            
            chapters.append({
                "chapter_id": f"chapter_{chapter_num}",
                "chapter_number": str(chapter_num),
                "title": f"Chapter {chapter_marker}",
                "text": chapter_text.strip()
            })
            chapter_num += 1
    
    return chapters


def extract_time_expressions(sentence_text: str, nlp) -> List[Dict[str, Any]]:
    """
    Extract time expressions from a sentence using spaCy NER and pattern matching.
    
    Args:
        sentence_text: Text of the sentence
        nlp: spaCy language model
        
    Returns:
        List of time expression dictionaries
    """
    doc = nlp(sentence_text)
    time_expressions = []
    
    # Extract DATE and TIME entities
    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME"]:
            time_expressions.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
    
    # Also check for temporal adverbs and phrases
    temporal_words = ["yesterday", "today", "tomorrow", "now", "then", "later", "earlier", 
                      "morning", "afternoon", "evening", "night", "noon", "midnight"]
    for token in doc:
        if token.text.lower() in temporal_words:
            # Check if not already captured as entity
            if not any(te["text"] == token.text for te in time_expressions):
                time_expressions.append({
                    "text": token.text,
                    "label": "TIME",
                    "start": token.idx,
                    "end": token.idx + len(token.text)
                })
    
    return time_expressions


def tokenize_sentences(chapter_text: str, nlp) -> List[Dict[str, Any]]:
    """
    Use spaCy to break chapter text into sentences.
    
    Args:
        chapter_text: Text content of a chapter
        nlp: spaCy language model
        
    Returns:
        List of sentence dictionaries with tokens and annotations
    """
    doc = nlp(chapter_text)
    sentences = []
    
    for sent_idx, sent in enumerate(doc.sents):
        sentence_data = {
            "sentence_id": f"sentence_{sent_idx + 1}",
            "text": sent.text.strip(),
            "tokens": [],
            "pos": [],
            "lemmas": [],
            "dependencies": [],
            "ner": [],
            "time_expressions": [],
            "event_ids": []  # Will be populated later in event extraction
        }
        
        # Extract tokens, POS tags, lemmas, and dependencies
        for token in sent:
            token_data = {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head": token.head.text,
                "head_pos": token.head.pos_,
                "is_punct": token.is_punct,
                "is_space": token.is_space
            }
            sentence_data["tokens"].append(token_data)
            sentence_data["pos"].append(token.pos_)
            sentence_data["lemmas"].append(token.lemma_)
            sentence_data["dependencies"].append({
                "token": token.text,
                "dep": token.dep_,
                "head": token.head.text
            })
        
        # Extract named entities (NER)
        for ent in sent.ents:
            sentence_data["ner"].append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char - sent.start_char,
                "end": ent.end_char - sent.start_char
            })
        
        # Extract time expressions
        sentence_data["time_expressions"] = extract_time_expressions(sent.text.strip(), nlp)
        
        sentences.append(sentence_data)
    
    return sentences


def annotate_linguistics(sentences: List[Dict], nlp) -> List[Dict]:
    """
    Extract POS tags, dependencies, and NER from sentences.
    Note: This is already done in tokenize_sentences, but kept for API consistency.
    
    Args:
        sentences: List of sentence dictionaries
        nlp: spaCy language model
        
    Returns:
        List of annotated sentence dictionaries
    """
    # Annotation is already performed in tokenize_sentences
    # This function is kept for API consistency and potential future enhancements
    return sentences


def process_text(input_file: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Main function that orchestrates all text processing.
    
    Args:
        input_file: Path to input text file
        output_dir: Output directory for processed files
        
    Returns:
        Dictionary containing chapters and sentences data
    """
    from .utils import read_text_file
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy English model not found. Please run: python -m spacy download en_core_web_sm"
        )
    
    # Read input text
    text = read_text_file(input_file)
    
    # Step 1: Segment chapters
    print("Segmenting chapters...")
    chapters = segment_chapters(text)
    #print(f"Found {len(chapters)} chapters")
    
    # Step 2: Tokenize sentences and annotate
    print("Tokenizing sentences and annotating...")
    all_sentences = []
    chapter_sentence_map = {}  # Map chapter_id to list of sentence IDs
    
    for chapter in chapters:
        chapter_id = chapter["chapter_id"]
        sentences = tokenize_sentences(chapter["text"], nlp)
        
        # Add chapter_id to each sentence and collect sentence IDs
        sentence_ids = []
        for sent in sentences:
            sent["chapter_id"] = chapter_id
            all_sentences.append(sent)
            sentence_ids.append(sent["sentence_id"])
        
        chapter_sentence_map[chapter_id] = sentence_ids
    
    #print(f"Processed {len(all_sentences)} sentences")
    
    # Prepare output data
    sentences_data = {
        "sentences": all_sentences,
        "total_sentences": len(all_sentences)
    }
    
    # Save outputs
    preprocessed_dir = f"{output_dir}/preprocessed"
    ensure_directory(preprocessed_dir)
    
    # Create chapters directory for individual chapter files
    chapters_dir = f"{preprocessed_dir}/chapters"
    ensure_directory(chapters_dir)
    
    # Save each chapter as its own JSON file with sentences and events arrays
    #print(f"Saving {len(chapters)} chapters as individual files...")
    for chapter in chapters:
        chapter_id = chapter["chapter_id"]
        # Add sentences and events arrays (events will be populated later)
        chapter_data = {
            "chapter_id": chapter_id,
            "title": chapter["title"],
            "text": chapter["text"],
            "sentences": chapter_sentence_map.get(chapter_id, []),
            "events": [],  # Will be populated in event extraction step
            "summary": None  # Optional, can be added later
        }
        chapter_file = f"{chapters_dir}/{chapter_id}.json"
        save_json(chapter_data, chapter_file)
        #print(f"  Saved {chapter_id}.json")
    
    # Also save a summary/index file for convenience with file paths
    chapters_index = {
        "total_chapters": len(chapters),
        "total_sentences": len(all_sentences),
        "chapter_ids": [ch["chapter_id"] for ch in chapters],
        "chapters": [
            {
                "chapter_id": ch["chapter_id"],
                "chapter_number": ch["chapter_number"],
                "title": ch["title"],
                "file": f"{ch['chapter_id']}.json"
            }
            for ch in chapters
        ]
    }
    save_json(chapters_index, f"{preprocessed_dir}/chapters_index.json")
    
    save_json(sentences_data, f"{preprocessed_dir}/sentences.json")
    
    #print(f"Saved {len(chapters)} chapters to {chapters_dir}/")
    #print(f"Saved chapters index to {preprocessed_dir}/chapters_index.json")
    #print(f"Saved sentences to {preprocessed_dir}/sentences.json")
    
    return {
        "chapters": chapters,
        "sentences": all_sentences
    }