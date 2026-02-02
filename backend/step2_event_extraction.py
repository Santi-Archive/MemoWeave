"""
Step 2: Event & Role Extraction
Transforms linguistic data into event frames using Semantic Role Labeling (SRL)
and dependency parsing.
"""

import spacy
from typing import List, Dict, Any, Optional
from transformers import pipeline, logging as hf_logging

# Silence HuggingFace messages about unused weights when loading models
hf_logging.set_verbosity_error()
from .utils import load_json, save_json, ensure_directory


def extract_events_with_srl(sentences: List[Dict], srl_model=None) -> List[Dict[str, Any]]:
    """
    Use HuggingFace SRL model to extract event frames.
    
    Args:
        sentences: List of sentence dictionaries from Step 1
        srl_model: Pre-loaded SRL pipeline (will create if None)
        
    Returns:
        List of event dictionaries extracted from SRL
    """
    if srl_model is None:
        # Use a BERT-based model for SRL
        # Note: We'll use a general model and extract predicates manually
        # For production, consider using a fine-tuned SRL model
        print("Loading SRL model...")
        try:
            from pathlib import Path
            import os
            
            # Try to load from local models directory first
            project_root = Path(__file__).parent.parent.absolute()
            local_model_path = project_root / "models" / "huggingface"
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            
            # Check if model files exist locally
            if local_model_path.exists() and any(local_model_path.iterdir()):
                # Check if key model files are present
                has_config = (local_model_path / "config.json").exists()
                has_model = any(local_model_path.glob("*.bin")) or any(local_model_path.glob("*.safetensors"))
                
                if has_config and has_model:
                    print(f"Loading from local models directory: {local_model_path}")
                    # Load directly from local path
                    srl_model = pipeline(
                        "token-classification",
                        model=str(local_model_path),
                        aggregation_strategy="simple"
                    )
                    print("SRL model loaded successfully from local directory")
                else:
                    # Model directory exists but incomplete, try with model name
                    print(f"Local model directory found but incomplete, loading from HuggingFace...")
                    # Set environment variables to use local cache
                    os.environ["TRANSFORMERS_CACHE"] = str(local_model_path)
                    os.environ["HF_HOME"] = str(local_model_path)
                    srl_model = pipeline(
                        "token-classification",
                        model=model_name,
                        aggregation_strategy="simple"
                    )
                    print("SRL model loaded successfully")
            else:
                # No local model, use system cache
                print("Loading from HuggingFace (will use system cache)...")
                srl_model = pipeline(
                    "token-classification",
                    model=model_name,
                    aggregation_strategy="simple"
                )
                print("SRL model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load SRL model: {e}")
            print("Falling back to dependency parsing only")
            print("Note: Event extraction will still work, but may be less accurate")
            srl_model = None
    
    events = []
    event_id_counter = 1
    
    for sentence in sentences:
        sentence_text = sentence["text"]
        chapter_id = sentence.get("chapter_id", "unknown")
        sentence_id = sentence.get("sentence_id", "unknown")
        
        # Extract predicates (verbs) from POS tags
        predicates = []
        for token in sentence.get("tokens", []):
            if token.get("pos") == "VERB" and not token.get("is_punct", False):
                predicates.append({
                    "text": token["text"],
                    "lemma": token.get("lemma", token["text"]),
                    "index": len(predicates)
                })
        
        # For each predicate, create an event frame
        for pred_idx, predicate in enumerate(predicates):
            # Get sentence text
            sentence_text = sentence.get("text", "")
            
            # Extract structured entities from sentence
            structured_entities = []
            for ent in sentence.get("ner", []):
                structured_entities.append({
                    "text": ent.get("text", ""),
                    "label": ent.get("label", ""),
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0)
                })
            
            event = {
                "event_id": f"event_{event_id_counter}",
                "chapter_id": chapter_id,
                "sentence_id": sentence_id,
                "text": sentence_text,  # Full sentence text
                "predicate": predicate["text"],  # Main verb/action
                "action": predicate["text"],  # Keep for backward compatibility
                "action_lemma": predicate["lemma"],
                "actor": None,  # Will be mapped to agent
                "target": None,  # Will be mapped to patient
                "location": None,  # ARGM-LOC
                "time_raw": None,  # ARGM-TMP
                "entities": structured_entities,  # Structured format
                "roles": {
                    "agent": None,  # ARG0
                    "patient": None,  # ARG1
                    "instrument": None,  # ARGM-MNR
                    "beneficiary": None,  # ARG2
                    "location": None,  # ARGM-LOC
                    "time": None  # ARGM-TMP
                },
                "time": {  # Embedded time object
                    "raw": None,
                    "normalized": None,
                    "type": None
                },
                "dependencies": sentence.get("dependencies", [])
            }
            
            # Try to extract roles using dependency parsing (fallback method)
            # This will be enhanced by fill_gaps_with_dependencies
            events.append(event)
            event_id_counter += 1
    
    return events


def fill_gaps_with_dependencies(events: List[Dict], sentences: List[Dict], nlp) -> List[Dict]:
    """
    Use spaCy dependencies to complete missing roles in events.
    
    Args:
        events: List of event dictionaries
        sentences: List of sentence dictionaries
        nlp: spaCy language model
        
    Returns:
        List of event dictionaries with filled roles
    """
    # Create a mapping from sentence_id to sentence data
    sentence_map = {sent["sentence_id"]: sent for sent in sentences}
    
    for event in events:
        sentence_id = event["sentence_id"]
        if sentence_id not in sentence_map:
            continue
        
        sentence = sentence_map[sentence_id]
        sentence_text = sentence["text"]
        
        # Process sentence with spaCy to get dependency tree
        doc = nlp(sentence_text)
        
        # Find the action verb in the sentence
        action = event.get("action")
        if not action:
            continue
        
        # Find the verb token
        verb_token = None
        for token in doc:
            if token.text == action or token.lemma_ == event.get("action_lemma", ""):
                if token.pos_ == "VERB":
                    verb_token = token
                    break
        
        if not verb_token:
            continue
        
        # Extract subject (ARG0 - agent/actor)
        if not event.get("actor") and not event["roles"].get("agent"):
            for child in verb_token.children:
                if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    # Get the full noun phrase
                    actor_text = " ".join([t.text for t in child.subtree])
                    event["actor"] = actor_text
                    event["roles"]["agent"] = actor_text
                    break
        
        # Extract direct object (ARG1 - patient/target)
        if not event.get("target") and not event["roles"].get("patient"):
            for child in verb_token.children:
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    target_text = " ".join([t.text for t in child.subtree])
                    event["target"] = target_text
                    event["roles"]["patient"] = target_text
                    break
        
        # Extract indirect object (ARG2 - beneficiary)
        if not event["roles"].get("beneficiary"):
            for child in verb_token.children:
                if child.dep_ in ["dative", "iobj"]:
                    beneficiary_text = " ".join([t.text for t in child.subtree])
                    event["roles"]["beneficiary"] = beneficiary_text
                    break
        
        # Extract instrument (ARGM-MNR)
        if not event["roles"].get("instrument"):
            for child in verb_token.children:
                if child.dep_ == "prep" and child.text.lower() in ["with", "using", "by"]:
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":
                            instrument_text = " ".join([t.text for t in prep_child.subtree])
                            event["roles"]["instrument"] = instrument_text
                            break
        
        # Extract location (ARGM-LOC)
        if not event.get("location") and not event["roles"].get("location"):
            for child in verb_token.children:
                if child.dep_ == "prep":
                    # Check if it's a location preposition
                    if child.text.lower() in ["in", "on", "at", "near", "by", "under", "over", "inside", "outside"]:
                        # Get the object of the preposition
                        for prep_child in child.children:
                            if prep_child.dep_ == "pobj":
                                location_text = " ".join([t.text for t in prep_child.subtree])
                                event["location"] = location_text
                                event["roles"]["location"] = location_text
                                break
        
        # Extract time (ARGM-TMP) - map to time role and time object
        if not event.get("time_raw") and not event["roles"].get("time"):
            # Check for temporal modifiers
            for child in verb_token.children:
                if child.dep_ == "prep":
                    if child.text.lower() in ["at", "on", "in", "during", "before", "after", "since", "until"]:
                        for prep_child in child.children:
                            if prep_child.dep_ == "pobj":
                                time_text = " ".join([t.text for t in prep_child.subtree])
                                event["time_raw"] = time_text
                                event["roles"]["time"] = time_text
                                event["time"]["raw"] = time_text
                                break
            
            # Also check for temporal adverbials
            if not event.get("time_raw"):
                for token in doc:
                    if token.dep_ == "advmod" and token.pos_ == "ADV":
                        # Check if it's a temporal adverb
                        temporal_words = ["yesterday", "today", "tomorrow", "now", "then", "later", "earlier"]
                        if token.text.lower() in temporal_words or any(tw in token.text.lower() for tw in temporal_words):
                            event["time_raw"] = token.text
                            event["roles"]["time"] = token.text
                            event["time"]["raw"] = token.text
                            break
        
        # Extract entities from sentence NER data (already structured)
        sentence_entities = sentence.get("ner", [])
        for sent_ent in sentence_entities:
            # Check if entity is not already in event entities
            entity_text = sent_ent.get("text", "")
            if not any(e.get("text") == entity_text for e in event.get("entities", [])):
                event["entities"].append({
                    "text": entity_text,
                    "label": sent_ent.get("label", ""),
                    "start": sent_ent.get("start", 0),
                    "end": sent_ent.get("end", 0)
                })
        
        # Also extract from spaCy doc for any missed entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "DATE", "TIME"]:
                entity_text = ent.text
                if not any(e.get("text") == entity_text for e in event.get("entities", [])):
                    event["entities"].append({
                        "text": entity_text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
    
    return events


def build_event_frames(sentences: List[Dict], srl_results: List[Dict], deps: List[Dict], nlp) -> List[Dict]:
    """
    Construct complete event frames from SRL results and dependencies.
    
    Args:
        sentences: List of sentence dictionaries
        srl_results: Results from SRL extraction
        deps: Dependency parsing results
        nlp: spaCy language model
        
    Returns:
        List of complete event frames
    """
    # Extract events using SRL
    events = extract_events_with_srl(sentences, srl_model=None)
    
    # Fill gaps using dependency parsing
    events = fill_gaps_with_dependencies(events, sentences, nlp)
    
    return events


def extract_events(input_dir: str, output_dir: str = "output") -> List[Dict[str, Any]]:
    """
    Main function to extract events from preprocessed sentences.
    
    Args:
        input_dir: Directory containing preprocessed files
        output_dir: Output directory for event files
        
    Returns:
        List of event dictionaries
    """
    from .utils import load_json
    
    # Load preprocessed data
    sentences_data = load_json(f"{input_dir}/preprocessed/sentences.json")
    sentences = sentences_data.get("sentences", [])
    
    #print(f"Extracting events from {len(sentences)} sentences...")
    print("Extracting events...")

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy English model not found. Please run: python -m spacy download en_core_web_sm"
        )
    
    # Extract events
    events = build_event_frames(sentences, [], [], nlp)
    
    #print(f"Extracted {len(events)} events")
    
    # Link events back to sentences (add event_ids to sentences)
    sentence_map = {sent["sentence_id"]: sent for sent in sentences}
    for event in events:
        sentence_id = event["sentence_id"]
        if sentence_id in sentence_map:
            if "event_ids" not in sentence_map[sentence_id]:
                sentence_map[sentence_id]["event_ids"] = []
            sentence_map[sentence_id]["event_ids"].append(event["event_id"])
    
    # Update sentences.json with event_ids
    sentences_data["sentences"] = list(sentence_map.values())
    save_json(sentences_data, f"{input_dir}/preprocessed/sentences.json")
    #print(f"Updated sentences with event_ids")
    
    # Link events back to chapters (add events arrays to chapter files)
    from pathlib import Path
    chapters_dir = Path(f"{input_dir}/preprocessed/chapters")
    if chapters_dir.exists():
        chapter_events_map = {}
        for event in events:
            chapter_id = event["chapter_id"]
            if chapter_id not in chapter_events_map:
                chapter_events_map[chapter_id] = []
            chapter_events_map[chapter_id].append(event["event_id"])
        
        # Update each chapter file
        for chapter_id, event_ids in chapter_events_map.items():
            chapter_file = chapters_dir / f"{chapter_id}.json"
            if chapter_file.exists():
                chapter_data = load_json(str(chapter_file))
                chapter_data["events"] = event_ids
                save_json(chapter_data, str(chapter_file))
        #print(f"Updated chapter files with events arrays")
    
    # Save events
    memory_dir = f"{output_dir}/memory"
    ensure_directory(memory_dir)
    
    events_data = {
        "events": events,
        "total_events": len(events)
    }
    
    save_json(events_data, f"{memory_dir}/events.json")
    #print(f"Saved events to {memory_dir}/events.json")
    
    return events