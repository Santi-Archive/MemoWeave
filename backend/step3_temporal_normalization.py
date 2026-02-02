"""
Step 3: Temporal Normalization
Converts vague or relative time phrases into standardized timestamps.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .utils import load_json, save_json, ensure_directory, get_reference_date

try:
    from heideltime import HeidelTime
except ImportError:
    HeidelTime = None
    print("Warning: python-heideltime not available. Using fallback normalization.")


def extract_time_expressions(events: List[Dict]) -> Dict[str, str]:
    """
    Collect all time expressions from events, mapped to event_id.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        Dictionary mapping event_id to time expression
    """
    event_time_map = {}
    
    for event in events:
        event_id = event.get("event_id")
        if not event_id:
            continue
        
        # Get time from time object
        time_obj = event.get("time", {})
        time_raw = time_obj.get("raw") or event.get("time_raw")
        
        # Get time from roles dictionary
        if not time_raw:
            roles = event.get("roles", {})
            time_raw = roles.get("time")
        
        # Get time from entities (DATE/TIME entities)
        if not time_raw:
            entities = event.get("entities", [])
            for entity in entities:
                if isinstance(entity, dict):
                    if entity.get("label") in ["DATE", "TIME"]:
                        time_raw = entity.get("text", "").strip()
                        break
        
        if time_raw:
            event_time_map[event_id] = time_raw.strip()
    
    return event_time_map


def normalize_with_heideltime(time_expr: str, reference_date: str) -> Dict[str, Any]:
    """
    Use HeidelTime to normalize time expressions.
    
    Args:
        time_expr: Time expression to normalize
        reference_date: Reference date in YYYY-MM-DD format
        
    Returns:
        Dictionary with normalized time information
    """
    if HeidelTime is None:
        # Fallback to custom normalization
        return normalize_time_fallback(time_expr, reference_date)
    
    try:
        # Initialize HeidelTime
        ht = HeidelTime()
        
        # Parse time expression
        # HeidelTime expects document text, so we'll create a simple document
        doc_text = f"Event happened {time_expr}."
        
        # Extract temporal expressions
        # Note: HeidelTime API may vary, this is a general approach
        result = ht.parse(doc_text, language='english', document_type='news')
        
        if result and len(result) > 0:
            # Extract normalized value from result
            normalized = result[0].get('value', time_expr)
            time_type = result[0].get('type', 'DATE')
            
            return {
                "original": time_expr,
                "normalized": normalized,
                "time_type": time_type,
                "confidence": 1.0
            }
    except Exception as e:
        print(f"Warning: HeidelTime normalization failed for '{time_expr}': {e}")
        return normalize_time_fallback(time_expr, reference_date)
    
    # Fallback if HeidelTime doesn't return results
    return normalize_time_fallback(time_expr, reference_date)


def normalize_time_fallback(time_expr: str, reference_date: str) -> Dict[str, Any]:
    """
    Fallback time normalization using regex and rule-based approach.
    
    Args:
        time_expr: Time expression to normalize
        reference_date: Reference date in YYYY-MM-DD format
        
    Returns:
        Dictionary with normalized time information
    """
    time_expr_lower = time_expr.lower().strip()
    ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
    
    # Absolute dates
    date_patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', 'DATE'),  # YYYY-MM-DD
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', 'DATE'),  # MM/DD/YYYY
        (r'(\d{1,2})-(\d{1,2})-(\d{4})', 'DATE'),  # MM-DD-YYYY
    ]
    
    for pattern, time_type in date_patterns:
        match = re.search(pattern, time_expr)
        if match:
            return {
                "original": time_expr,
                "normalized": time_expr,
                "time_type": time_type,
                "confidence": 0.9
            }
    
    # Relative times
    relative_patterns = {
        r'yesterday': lambda d: (d - timedelta(days=1)).strftime("%Y-%m-%d"),
        r'today': lambda d: d.strftime("%Y-%m-%d"),
        r'tomorrow': lambda d: (d + timedelta(days=1)).strftime("%Y-%m-%d"),
        r'(\d+)\s*days?\s*later': lambda d, m: (d + timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*days?\s*ago': lambda d, m: (d - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*weeks?\s*later': lambda d, m: (d + timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*weeks?\s*ago': lambda d, m: (d - timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*months?\s*later': lambda d, m: (d + timedelta(days=30*int(m.group(1)))).strftime("%Y-%m-%d"),
        r'(\d+)\s*months?\s*ago': lambda d, m: (d - timedelta(days=30*int(m.group(1)))).strftime("%Y-%m-%d"),
    }
    
    for pattern, func in relative_patterns.items():
        match = re.search(pattern, time_expr_lower)
        if match:
            try:
                if match.groups():
                    normalized = func(ref_date, match)
                else:
                    normalized = func(ref_date)
                return {
                    "original": time_expr,
                    "normalized": normalized,
                    "time_type": "DATE",
                    "confidence": 0.8
                }
            except Exception:
                pass
    
    # Time of day
    time_of_day = {
        'morning': 'T-MORNING',
        'afternoon': 'T-AFTERNOON',
        'evening': 'T-EVENING',
        'night': 'T-NIGHT',
        'noon': 'T-NOON',
        'midnight': 'T-MIDNIGHT'
    }
    
    for key, value in time_of_day.items():
        if key in time_expr_lower:
            return {
                "original": time_expr,
                "normalized": value,
                "time_type": "TIME",
                "confidence": 0.7
            }
    
    # Relative placeholders for vague times
    vague_patterns = {
        r'(\d+)\s*days?\s*later': 'REL-{}D',
        r'(\d+)\s*weeks?\s*later': 'REL-{}W',
        r'(\d+)\s*months?\s*later': 'REL-{}M',
        r'later': 'REL-LATER',
        r'soon': 'REL-SOON',
        r'eventually': 'REL-EVENTUALLY',
        r'eventually': 'REL-EVENTUALLY',
    }
    
    for pattern, placeholder in vague_patterns.items():
        match = re.search(pattern, time_expr_lower)
        if match:
            if match.groups():
                num = match.group(1)
                normalized = placeholder.format(num)
            else:
                normalized = placeholder
            return {
                "original": time_expr,
                "normalized": normalized,
                "time_type": "RELATIVE",
                "confidence": 0.6
            }
    
    # Check for common temporal phrases that might be missed
    temporal_phrases = {
        'next week': lambda d: (d + timedelta(weeks=1)).strftime("%Y-%m-%d"),
        'last week': lambda d: (d - timedelta(weeks=1)).strftime("%Y-%m-%d"),
        'next month': lambda d: (d + timedelta(days=30)).strftime("%Y-%m-%d"),
        'last month': lambda d: (d - timedelta(days=30)).strftime("%Y-%m-%d"),
        'next year': lambda d: (d + timedelta(days=365)).strftime("%Y-%m-%d"),
        'last year': lambda d: (d - timedelta(days=365)).strftime("%Y-%m-%d"),
    }
    
    for phrase, func in temporal_phrases.items():
        if phrase in time_expr_lower:
            try:
                normalized = func(ref_date)
                return {
                    "original": time_expr,
                    "normalized": normalized,
                    "time_type": "DATE",
                    "confidence": 0.75
                }
            except Exception:
                pass
    
    # Check for numeric patterns that might be dates
    # Try to match patterns like "January 15" or "Jan 15"
    month_names = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
        'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
        'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    
    for month_name, month_num in month_names.items():
        pattern = rf'{month_name}\s+(\d{{1,2}})(?:\s*,?\s*(\d{{4}}))?'
        match = re.search(pattern, time_expr_lower)
        if match:
            day = int(match.group(1))
            year = int(match.group(2)) if match.group(2) else ref_date.year
            try:
                normalized = datetime(year, month_num, day).strftime("%Y-%m-%d")
                return {
                    "original": time_expr,
                    "normalized": normalized,
                    "time_type": "DATE",
                    "confidence": 0.8
                }
            except ValueError:
                pass
    
    # Default: keep original with low confidence (only if truly unrecognizable)
    # Try to avoid UNKNOWN by checking if it looks like it might be a time expression
    if any(word in time_expr_lower for word in ['time', 'day', 'week', 'month', 'year', 'hour', 'minute']):
        return {
            "original": time_expr,
            "normalized": time_expr,
            "time_type": "RELATIVE",
            "confidence": 0.4
        }
    
    return {
        "original": time_expr,
        "normalized": time_expr,
        "time_type": "UNKNOWN",
        "confidence": 0.3
    }


def attach_normalized_times(events: List[Dict], event_time_map: Dict[str, str], normalized_times: Dict[str, Dict]) -> List[Dict]:
    """
    Update event frames with normalized timestamps, embedding time object directly.
    
    Args:
        events: List of event dictionaries
        event_time_map: Dictionary mapping event_id to time expression
        normalized_times: Dictionary mapping time expressions to normalized data
        
    Returns:
        List of events with normalized time fields embedded in time object
    """
    for event in events:
        event_id = event.get("event_id")
        if not event_id:
            continue
        
        # Get time expression for this event
        time_expr = event_time_map.get(event_id)
        
        # Initialize time object if not present
        if "time" not in event:
            event["time"] = {"raw": None, "normalized": None, "type": None}
        
        if time_expr and time_expr in normalized_times:
            normalized_data = normalized_times[time_expr]
            # Embed time object directly in event
            event["time"]["raw"] = normalized_data["original"]
            event["time"]["normalized"] = normalized_data["normalized"]
            event["time"]["type"] = normalized_data["time_type"]
            
            # Also keep backward compatibility fields
            event["time_raw"] = normalized_data["original"]
            event["time_normalized"] = normalized_data["normalized"]
            event["time_type"] = normalized_data["time_type"]
            event["time_confidence"] = normalized_data.get("confidence", 0.5)
            
            # Update roles time field
            if "roles" in event and "time" in event["roles"]:
                event["roles"]["time"] = normalized_data["original"]
        else:
            # No time expression found, ensure time object is initialized
            if event["time"]["raw"] is None:
                event["time"]["raw"] = None
                event["time"]["normalized"] = None
                event["time"]["type"] = None
            
            # Set backward compatibility fields
            event["time_normalized"] = None
            event["time_type"] = None
    
    return events


def normalize_temporal_expressions(input_dir: str, output_dir: str = "output", reference_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to normalize temporal expressions in events.
    
    Args:
        input_dir: Directory containing event files
        output_dir: Output directory for timestamp files
        reference_date: Reference date for normalization (defaults to current date)
        
    Returns:
        Dictionary containing normalized timestamps
    """
    from .utils import load_json
    
    # Load events
    events_data = load_json(f"{input_dir}/memory/events.json")
    events = events_data.get("events", [])
    
    #print(f"Normalizing temporal expressions for {len(events)} events...")
    
    # Get reference date
    if reference_date is None:
        reference_date = get_reference_date()
    
    # Extract time expressions mapped to event_id
    event_time_map = extract_time_expressions(events)
    print(f"Found {len(event_time_map)} events with time expressions")
    
    # Get unique time expressions for normalization
    unique_time_expressions = set(event_time_map.values())
    print(f"Found {len(unique_time_expressions)} unique time expressions")
    
    # Normalize each unique time expression
    normalized_times = {}
    for time_expr in unique_time_expressions:
        normalized = normalize_with_heideltime(time_expr, reference_date)
        normalized_times[time_expr] = normalized
    
    # Attach normalized times to events (linked by event_id)
    events = attach_normalized_times(events, event_time_map, normalized_times)
    
    # Prepare timestamp data (for debugging)
    timestamps_data = {
        "reference_date": reference_date,
        "normalized_times": normalized_times,
        "total_expressions": len(unique_time_expressions)
    }
    
    # Prepare event_timestamps data (event_id -> time mapping)
    event_timestamps = []
    for event in events:
        event_id = event.get("event_id")
        time_obj = event.get("time", {})
        if time_obj.get("raw"):
            event_timestamps.append({
                "event_id": event_id,
                "raw_time": time_obj.get("raw"),
                "normalized_time": time_obj.get("normalized"),
                "time_type": time_obj.get("type")
            })
    
    # Save timestamps (debug file)
    memory_dir = f"{output_dir}/memory"
    ensure_directory(memory_dir)
    
    save_json(timestamps_data, f"{memory_dir}/timestamps.json")
    print(f"Saved timestamps to {memory_dir}/timestamps.json")
    
    # Save event_timestamps (optional, for debugging)
    event_timestamps_data = {
        "reference_date": reference_date,
        "event_timestamps": event_timestamps,
        "total_events_with_time": len(event_timestamps)
    }
    save_json(event_timestamps_data, f"{memory_dir}/event_timestamps.json")
    print(f"Saved event_timestamps to {memory_dir}/event_timestamps.json")
    
    # Update and save events with normalized times (embedded in time object)
    events_data["events"] = events
    save_json(events_data, f"{memory_dir}/events.json")
    #print(f"Updated events with normalized times (embedded in time object)")
    
    return timestamps_data