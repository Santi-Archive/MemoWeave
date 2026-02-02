"""
Step 5: Temporal Memory Storage Layer
Constructs the final, unified memory module combining all processed data.
"""

from typing import List, Dict, Any, Set
from pathlib import Path
from datetime import datetime
from .utils import load_json, save_json, ensure_directory


def extract_characters_entities(events: List[Dict], sentences: List[Dict]) -> Dict[str, Any]:
    """
    Extract unique characters and entities from events and sentences.
    
    Args:
        events: List of event dictionaries
        sentences: List of sentence dictionaries
        
    Returns:
        Dictionary containing characters and entities
    """
    characters: Set[str] = set()
    locations: Set[str] = set()
    organizations: Set[str] = set()
    dates: Set[str] = set()
    times: Set[str] = set()
    other_entities: Set[str] = set()
    
    # Extract from events
    for event in events:
        # Extract actor (likely a character)
        actor = event.get("actor")
        if actor:
            characters.add(actor.strip())
        
        # Extract entities from event
        entities = event.get("entities", [])
        for entity in entities:
            if isinstance(entity, str):
                # Simple string entity
                other_entities.add(entity.strip())
            elif isinstance(entity, dict):
                # Structured entity with label
                entity_text = entity.get("text", "").strip()
                entity_label = entity.get("label", "").upper()
                
                if entity_label == "PERSON":
                    characters.add(entity_text)
                elif entity_label in ["GPE", "LOC"]:
                    locations.add(entity_text)
                elif entity_label == "ORG":
                    organizations.add(entity_text)
                elif entity_label == "DATE":
                    dates.add(entity_text)
                elif entity_label == "TIME":
                    times.add(entity_text)
                else:
                    other_entities.add(entity_text)
    
    # Extract from sentences
    for sentence in sentences:
        entities = sentence.get("entities", [])
        for entity in entities:
            if isinstance(entity, dict):
                entity_text = entity.get("text", "").strip()
                entity_label = entity.get("label", "").upper()
                
                if entity_label == "PERSON":
                    characters.add(entity_text)
                elif entity_label in ["GPE", "LOC"]:
                    locations.add(entity_text)
                elif entity_label == "ORG":
                    organizations.add(entity_text)
                elif entity_label == "DATE":
                    dates.add(entity_text)
                elif entity_label == "TIME":
                    times.add(entity_text)
                else:
                    other_entities.add(entity_text)
    
    return {
        "characters": sorted(list(characters)),
        "locations": sorted(list(locations)),
        "organizations": sorted(list(organizations)),
        "dates": sorted(list(dates)),
        "times": sorted(list(times)),
        "other_entities": sorted(list(other_entities)),
        "total_characters": len(characters),
        "total_locations": len(locations),
        "total_organizations": len(organizations)
    }


def build_timeline(events: List[Dict]) -> List[str]:
    """
    Build timeline of events sorted by normalized time.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        List of event_ids sorted by normalized_time
    """
    # Create list of (event_id, normalized_time) tuples
    event_times = []
    for event in events:
        event_id = event.get("event_id")
        time_obj = event.get("time", {})
        normalized_time = time_obj.get("normalized") or event.get("time_normalized")
        
        if normalized_time:
            # Try to parse as date for sorting
            try:
                # Try different date formats
                for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        parsed_time = datetime.strptime(normalized_time, fmt)
                        event_times.append((event_id, parsed_time, normalized_time))
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matches, use string comparison
                    event_times.append((event_id, None, normalized_time))
            except Exception:
                event_times.append((event_id, None, normalized_time))
        else:
            # Events without time go to the end
            event_times.append((event_id, None, None))
    
    # Sort by parsed time, then by string
    event_times.sort(key=lambda x: (x[1] is not None, x[1] if x[1] is not None else "", x[2] or ""))
    
    return [event_id for event_id, _, _ in event_times]


def build_temporal_edges(events: List[Dict]) -> List[Dict[str, Any]]:
    """
    Build temporal edges (before/after relations) between events.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        List of temporal edge dictionaries
    """
    temporal_edges = []
    event_map = {event.get("event_id"): event for event in events}
    
    # Build timeline to get ordering
    timeline = build_timeline(events)
    
    # Create edges based on timeline ordering
    for i in range(len(timeline) - 1):
        from_event_id = timeline[i]
        to_event_id = timeline[i + 1]
        
        from_event = event_map.get(from_event_id, {})
        to_event = event_map.get(to_event_id, {})
        
        from_time = from_event.get("time", {}).get("normalized") or from_event.get("time_normalized")
        to_time = to_event.get("time", {}).get("normalized") or to_event.get("time_normalized")
        
        if from_time and to_time:
            temporal_edges.append({
                "from_event_id": from_event_id,
                "to_event_id": to_event_id,
                "relation": "before",
                "confidence": 0.9  # High confidence for timeline-based ordering
            })
    
    return temporal_edges


def build_semantic_edges(semantic_memory: List[Dict], similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Build semantic edges from semantic memory neighbors.
    
    Args:
        semantic_memory: List of semantic memory entries
        similarity_threshold: Minimum similarity for edge creation
        
    Returns:
        List of semantic edge dictionaries
    """
    semantic_edges = []
    
    for entry in semantic_memory:
        event_id = entry.get("event_id")
        neighbors = entry.get("semantic_neighbors", [])
        
        for neighbor in neighbors:
            neighbor_id = neighbor.get("event_id")
            similarity = neighbor.get("similarity", 0.0)
            
            if similarity >= similarity_threshold:
                semantic_edges.append({
                    "from_event_id": event_id,
                    "to_event_id": neighbor_id,
                    "similarity": similarity,
                    "threshold": similarity_threshold
                })
    
    return semantic_edges


def build_event_graph(temporal_edges: List[Dict], semantic_edges: List[Dict]) -> Dict[str, Any]:
    """
    Build unified event graph combining temporal and semantic edges.
    
    Args:
        temporal_edges: List of temporal edge dictionaries
        semantic_edges: List of semantic edge dictionaries
        
    Returns:
        Event graph structure
    """
    # Build adjacency lists
    temporal_graph = {}
    semantic_graph = {}
    
    for edge in temporal_edges:
        from_id = edge["from_event_id"]
        if from_id not in temporal_graph:
            temporal_graph[from_id] = []
        temporal_graph[from_id].append({
            "to_event_id": edge["to_event_id"],
            "relation": edge["relation"],
            "confidence": edge.get("confidence", 0.5)
        })
    
    for edge in semantic_edges:
        from_id = edge["from_event_id"]
        if from_id not in semantic_graph:
            semantic_graph[from_id] = []
        semantic_graph[from_id].append({
            "to_event_id": edge["to_event_id"],
            "similarity": edge["similarity"]
        })
    
    return {
        "temporal_edges": temporal_graph,
        "semantic_edges": semantic_graph,
        "total_temporal_edges": len(temporal_edges),
        "total_semantic_edges": len(semantic_edges)
    }


def build_chapter_map(events: List[Dict]) -> Dict[str, List[str]]:
    """
    Build chapter map: chapter_id -> list of event_ids.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        Dictionary mapping chapter_id to list of event_ids
    """
    chapter_map = {}
    
    for event in events:
        chapter_id = event.get("chapter_id")
        event_id = event.get("event_id")
        
        if chapter_id and event_id:
            if chapter_id not in chapter_map:
                chapter_map[chapter_id] = []
            chapter_map[chapter_id].append(event_id)
    
    return chapter_map


def build_canonical_entity_graph(events: List[Dict], sentences: List[Dict]) -> Dict[str, Any]:
    """
    Build canonical entity graph from events and sentences.
    
    Args:
        events: List of event dictionaries
        sentences: List of sentence dictionaries
        
    Returns:
        Canonical entity graph structure
    """
    # Extract all entities with their types
    entity_map = {}  # entity_text -> {label, occurrences: [event_ids]}
    
    for event in events:
        event_id = event.get("event_id")
        entities = event.get("entities", [])
        
        for entity in entities:
            if isinstance(entity, dict):
                entity_text = entity.get("text", "")
                entity_label = entity.get("label", "")
                
                if entity_text:
                    if entity_text not in entity_map:
                        entity_map[entity_text] = {
                            "label": entity_label,
                            "occurrences": []
                        }
                    entity_map[entity_text]["occurrences"].append(event_id)
    
    # Group by label
    entities_by_label = {}
    for entity_text, entity_data in entity_map.items():
        label = entity_data["label"]
        if label not in entities_by_label:
            entities_by_label[label] = []
        entities_by_label[label].append({
            "text": entity_text,
            "occurrences": entity_data["occurrences"],
            "frequency": len(entity_data["occurrences"])
        })
    
    return {
        "entities": entity_map,
        "by_label": entities_by_label,
        "total_unique_entities": len(entity_map)
    }


def build_memory_module(
    chapters: List[Dict],
    sentences: List[Dict],
    events: List[Dict],
    timestamps: Dict[str, Any],
    embeddings: Dict[str, Any],
    entities: Dict[str, Any],
    semantic_memory: List[Dict]
) -> Dict[str, Any]:
    """
    Combine all data into unified memory module.
    
    Args:
        chapters: List of chapter dictionaries
        sentences: List of sentence dictionaries
        events: List of event dictionaries
        timestamps: Dictionary containing timestamp normalization data
        embeddings: Dictionary containing embedding data
        entities: Dictionary containing extracted entities
        semantic_memory: List of semantic memory entries
        
    Returns:
        Unified memory module dictionary
    """
    # Build timeline
    print("Building timeline...")
    timeline = build_timeline(events)
    
    # Build temporal edges
    print("Building temporal edges...")
    temporal_edges = build_temporal_edges(events)
    
    # Build semantic edges
    print("Building semantic edges...")
    semantic_edges = build_semantic_edges(semantic_memory, similarity_threshold=0.7)
    
    # Build event graph
    print("Building event graph...")
    event_graph = build_event_graph(temporal_edges, semantic_edges)
    
    # Build chapter map
    print("Building chapter map...")
    chapter_map = build_chapter_map(events)
    
    # Build canonical entity graph
    print("Building canonical entity graph...")
    canonical_entities = build_canonical_entity_graph(events, sentences)
    
    memory_module = {
        "events": events,  # Full event frames
        "entities": canonical_entities,  # Canonical entity graph
        "semantic_memory": semantic_memory,  # Embeddings + similarity
        "timeline": timeline,  # Sorted by normalized time
        "temporal_edges": temporal_edges,  # Before/after relations
        "semantic_edges": semantic_edges,  # Semantic similarity edges
        "chapter_map": chapter_map,  # Chapter → event IDs
        "event_graph": event_graph,  # Semantic + temporal edges
        "metadata": {
            "generated_on": datetime.now().strftime("%Y-%m-%d"),
            "model": embeddings.get("model_name", "unknown"),
            "total_chapters": len(chapters),
            "total_sentences": len(sentences),
            "total_events": len(events),
            "total_temporal_edges": len(temporal_edges),
            "total_semantic_edges": len(semantic_edges),
            "total_characters": entities.get("total_characters", 0),
            "total_locations": entities.get("total_locations", 0),
            "total_organizations": entities.get("total_organizations", 0),
            "embedding_dim": embeddings.get("embedding_dim", 0),
            "embedding_model": embeddings.get("model_name", "unknown")
        }
    }
    
    return memory_module


def save_memory_module(memory_module: Dict[str, Any], output_path: str) -> None:
    """
    Save final JSON memory module.
    
    Args:
        memory_module: Unified memory module dictionary
        output_path: Path where to save the memory module
    """
    ensure_directory(output_path.rsplit('/', 1)[0] if '/' in output_path else output_path.rsplit('\\', 1)[0])
    save_json(memory_module, output_path)
    print(f"Saved unified memory module to {output_path}")


def create_memory_module(input_dir: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Main function to create the unified memory module.
    
    Args:
        input_dir: Directory containing all processed files
        output_dir: Output directory for memory module
        
    Returns:
        Unified memory module dictionary
    """
    print("Building unified memory module...")
    
    # Load all processed data
    print("Loading preprocessed data...")
    
    # Load chapters from individual files
    chapters_dir = Path(f"{input_dir}/preprocessed/chapters")
    chapters = []
    
    if chapters_dir.exists():
        # Load all chapter JSON files
        chapter_files = sorted(chapters_dir.glob("chapter_*.json"))
        print(f"Loading {len(chapter_files)} chapter files...")
        for chapter_file in chapter_files:
            chapter_data = load_json(str(chapter_file))
            chapters.append(chapter_data)
    else:
        # Fallback: try to load from old chapters.json format
        print("Warning: chapters directory not found, trying old format...")
        try:
            chapters_data = load_json(f"{input_dir}/preprocessed/chapters.json")
            chapters = chapters_data.get("chapters", [])
        except FileNotFoundError:
            print("Error: No chapters found in either format")
            chapters = []
    
    sentences_data = load_json(f"{input_dir}/preprocessed/sentences.json")
    sentences = sentences_data.get("sentences", [])
    
    events_data = load_json(f"{input_dir}/memory/events.json")
    events = events_data.get("events", [])
    
    timestamps_data = load_json(f"{input_dir}/memory/timestamps.json")
    
    embeddings_data = load_json(f"{input_dir}/memory/event_embeddings.json")
    
    # Load semantic memory
    semantic_memory_data = load_json(f"{input_dir}/memory/memory_semantic.json")
    semantic_memory = semantic_memory_data.get("semantic_memory", [])
    
    # Extract characters and entities
    print("Extracting characters and entities...")
    entities = extract_characters_entities(events, sentences)
    
    # Build unified memory module
    print("Combining all data into memory module...")
    memory_module = build_memory_module(
        chapters=chapters,
        sentences=sentences,
        events=events,
        timestamps=timestamps_data,
        embeddings=embeddings_data,
        entities=entities,
        semantic_memory=semantic_memory
    )
    
    # Save memory module
    memory_dir = f"{output_dir}/memory"
    ensure_directory(memory_dir)
    
    save_memory_module(memory_module, f"{memory_dir}/memory_module.json")
    
    print(f"Memory module created successfully!")
    print(f"  - {len(chapters)} chapters")
    print(f"  - {len(sentences)} sentences")
    print(f"  - {len(events)} events")
    print(f"  - {entities.get('total_characters', 0)} characters")
    print(f"  - {entities.get('total_locations', 0)} locations")
    
    return memory_module