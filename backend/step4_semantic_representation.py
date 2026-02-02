"""
Step 4: Semantic Representation & Memory Structuring
Converts symbolic events into meaningful numerical embeddings.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
#from sentence_transformers import SentenceTransformer
from .utils import load_json, save_json, ensure_directory


def format_event_string(event: Dict[str, Any]) -> str:
    """
    Create semantic string representation of an event.
    
    Args:
        event: Event dictionary
        
    Returns:
        Formatted semantic string
    """
    parts = []
    
    # Actor
    actor = event.get("actor")
    if actor:
        parts.append(f"Actor: {actor}")
    
    # Action
    action = event.get("action")
    if action:
        parts.append(f"Action: {action}")
    
    # Target
    target = event.get("target")
    if target:
        parts.append(f"Target: {target}")
    
    # Location
    location = event.get("location")
    if location:
        parts.append(f"Location: {location}")
    
    # Time
    time_normalized = event.get("time_normalized")
    if time_normalized:
        parts.append(f"Time: {time_normalized}")
    else:
        time_raw = event.get("time_raw")
        if time_raw:
            parts.append(f"Time: {time_raw}")
    
    # If no parts, create a minimal representation
    if not parts:
        parts.append(f"Event: {action or 'unknown'}")
    
    return "; ".join(parts) + "."


def generate_embeddings(
    event_strings: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Use sentence transformers to create embeddings.
    Safe for web servers, subprocesses, and PyInstaller.
    """
    import os
    import sys
    from pathlib import Path

    # ---- CRITICAL FIX ----
    # Disable tqdm globally BEFORE importing sentence_transformers
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Monkey-patch tqdm to never call isatty
    try:
        import tqdm
        tqdm.tqdm = lambda *args, **kwargs: args[0]
    except Exception:
        pass
    # ----------------------

    from sentence_transformers import SentenceTransformer

    # Resolve project root
    project_root = Path(__file__).parent.parent.absolute()
    local_cache = project_root / "models" / "sentence_transformers"

    # Load model
    if local_cache.exists():
        model = SentenceTransformer(
            model_name,
            cache_folder=str(local_cache)
        )
    else:
        model = SentenceTransformer(model_name)

    # Encode WITHOUT progress bars
    embeddings = model.encode(
        event_strings,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embeddings


def compute_semantic_neighbors(embeddings: np.ndarray, event_ids: List[str], top_k: int = 10, similarity_threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
    """
    Compute semantic neighbors for each event using cosine similarity.
    
    Args:
        embeddings: NumPy array of embedding vectors
        event_ids: List of event IDs corresponding to embeddings
        top_k: Number of top neighbors to return
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        List of lists, where each inner list contains neighbor dictionaries
    """
    # Try to use sklearn, fallback to numpy implementation
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
    except ImportError:
        # Fallback: compute cosine similarity using numpy
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings / norms
        
        # Compute cosine similarity: dot product of normalized vectors
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    neighbors_list = []
    for i in range(len(embeddings)):
        # Get similarities for this event (excluding self)
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build neighbor list
        neighbors = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= similarity_threshold:
                neighbors.append({
                    "event_id": event_ids[idx] if idx < len(event_ids) else f"event_{idx + 1}",
                    "similarity": round(similarity, 4)
                })
        
        neighbors_list.append(neighbors)
    
    return neighbors_list


def build_semantic_memory(events: List[Dict], embeddings: np.ndarray) -> pd.DataFrame:
    """
    Create semantic memory table with event data and embeddings.
    
    Args:
        events: List of event dictionaries
        embeddings: NumPy array of embedding vectors
        
    Returns:
        Pandas DataFrame containing semantic memory
    """
    rows = []
    
    # Get event IDs list
    event_ids = [event.get("event_id", f"event_{i}") for i, event in enumerate(events)]
    
    # Compute semantic neighbors
    print("Computing semantic neighbors...")
    neighbors_list = compute_semantic_neighbors(embeddings, event_ids, top_k=10, similarity_threshold=0.0)
    
    for idx, event in enumerate(events):
        # Format semantic string
        semantic_string = format_event_string(event)
        
        # Get embedding vector
        embedding_vector = embeddings[idx].tolist() if idx < len(embeddings) else []
        
        # Get event text
        event_text = event.get("text", "")
        
        # Get entities (structured)
        entities = event.get("entities", [])
        
        # Get normalized time from time object
        time_obj = event.get("time", {})
        normalized_time = time_obj.get("normalized") or event.get("time_normalized")
        
        # Get neighbors for this event
        neighbors = neighbors_list[idx]
        
        row = {
            "event_id": event.get("event_id", f"event_{idx}"),
            "chapter_id": event.get("chapter_id", "unknown"),
            "sentence_id": event.get("sentence_id", "unknown"),
            "text": event_text,  # Full event text
            "entities": entities,  # Structured entities
            "normalized_time": normalized_time,  # Normalized timestamp
            "semantic_string": semantic_string,
            "normalized_timestamp": normalized_time,  # Backward compatibility
            "timestamp_type": time_obj.get("type") or event.get("time_type"),
            "embedding_vector": embedding_vector,
            "embedding_dim": len(embedding_vector),
            "semantic_neighbors": neighbors  # Top-k similar events
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def create_semantic_representations(input_dir: str, output_dir: str = "output", model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Main function to create semantic representations and embeddings.
    
    Args:
        input_dir: Directory containing event files
        output_dir: Output directory for semantic memory files
        model_name: Name of sentence transformer model to use
        
    Returns:
        Dictionary containing semantic memory data
    """
    from .utils import load_json
    
    # Load events
    events_data = load_json(f"{input_dir}/memory/events.json")
    events = events_data.get("events", [])
    
    print(f"Creating semantic representations for {len(events)} events...")
    
    # Format event strings
    print("Formatting event strings...")
    event_strings = [format_event_string(event) for event in events]
    
    # Generate embeddings
    embeddings = generate_embeddings(event_strings, model_name)
    
    # Build semantic memory DataFrame
    print("Building semantic memory table...")
    semantic_memory_df = build_semantic_memory(events, embeddings)
    
    # Convert DataFrame to list of dictionaries for JSON serialization
    semantic_memory_list = semantic_memory_df.to_dict(orient='records')
    
    # Restructure to match thesis requirements
    # Each entry should have: event_id, embedding, text, entities, normalized_time, chapter_id, sentence_id, semantic_neighbors
    restructured_memory = []
    for record in semantic_memory_list:
        restructured_entry = {
            "event_id": record["event_id"],
            "embedding": record["embedding_vector"],
            "text": record["text"],
            "entities": record["entities"],
            "normalized_time": record["normalized_time"],
            "chapter_id": record["chapter_id"],
            "sentence_id": record["sentence_id"],
            "semantic_neighbors": record["semantic_neighbors"]
        }
        restructured_memory.append(restructured_entry)
    
    # Get embedding dimension
    embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])
    
    # Prepare semantic memory data (thesis-aligned structure)
    memory_data = {
        "model": model_name,
        "dim": embedding_dim,
        "semantic_memory": restructured_memory
    }
    
    # Prepare embedding data (store separately for efficiency/debugging)
    embeddings_data = {
        "event_ids": [event.get("event_id", f"event_{i}") for i, event in enumerate(events)],
        "embeddings": embeddings.tolist(),
        "embedding_dim": embedding_dim,
        "model_name": model_name
    }
    
    # Save outputs
    memory_dir = f"{output_dir}/memory"
    ensure_directory(memory_dir)
    
    save_json(embeddings_data, f"{memory_dir}/event_embeddings.json")
    #print(f"Saved embeddings to {memory_dir}/event_embeddings.json")
    
    save_json(memory_data, f"{memory_dir}/memory_semantic.json")
    #print(f"Saved semantic memory to {memory_dir}/memory_semantic.json")
    
    return {
        "semantic_memory": memory_data,
        "embeddings": embeddings_data
    }