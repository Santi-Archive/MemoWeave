# character.py

import os
import csv
import json
import requests
from dotenv import load_dotenv
from typing import Callable, List, Dict, Optional

load_dotenv()

# =========================
# Configuration
# =========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"

def log(msg: str, callback: Callable = None):
    formatted = msg
    if callback:
        callback(formatted)
    else:
        print(formatted)

# =========================
# Helper Functions
# =========================

def load_reasoning_graph(output_dir: str = "output") -> Optional[Dict]:
    """
    Load reasoning graph JSON if it exists.
    Returns None if file doesn't exist (safe fallback).
    """
    reasoning_path = os.path.join(output_dir, "memory", "reasoning_graph.json")
    if os.path.exists(reasoning_path):
        try:
            with open(reasoning_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load reasoning graph: {e}")
            return None
    return None

def format_reasoning_graph(reasoning_graph: Optional[Dict]) -> str:
    """
    Convert reasoning graph JSON to readable text for LLM.
    Returns empty string if None passed (safe fallback).
    """
    if not reasoning_graph:
        return ""
    
    text = "\n### TEMPORAL & CAUSAL RELATIONS ###\n\n"
    
    # Format temporal relations
    temporal_rels = reasoning_graph.get("temporal_relations", [])
    if temporal_rels:
        text += "**Timeline (Temporal Order):**\n"
        for rel in temporal_rels:
            from_evt = rel.get("from_event", "unknown")
            to_evt = rel.get("to_event", "unknown")
            relation = rel.get("relation", "RELATED")
            text += f"• Event '{from_evt}' happens {relation} Event '{to_evt}'\n"
        text += "\n"
    
    # Format causal relations
    causal_rels = reasoning_graph.get("causal_relations", [])
    if causal_rels:
        text += "**Cause & Effect Relationships:**\n"
        for rel in causal_rels:
            from_evt = rel.get("from_event", "unknown")
            to_evt = rel.get("to_event", "unknown")
            relation = rel.get("relation", "AFFECTS")
            text += f"• Event '{from_evt}' {relation} Event '{to_evt}'\n"
        text += "\n"
    
    return text

# =========================
# CSV and Prompt Functions
# =========================

def read_csv_as_chapter_text(csv_path: str) -> Dict[str, List[str]]:
    """
    Reads role_completeness.csv and aggregates events per chapter
    into semi-narrative strings for LLM.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    chapters = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chapter_id = row.get("chapter_id", "unknown")
            actor = row.get("actor") or "Unknown actor"
            target = row.get("target") or "Unknown target"
            location = row.get("location") or "Unknown location"
            text = f"- {row.get('text', '')} (actor: {actor}, target: {target}, location: {location})"
            chapters.setdefault(chapter_id, []).append(text)

    return chapters

def build_prompt(chapters: Dict[str, List[str]], reasoning_graph: Optional[Dict] = None) -> str:
    """
    Builds a single macro-level prompt for the LLM using chapter aggregation.
    Optionally includes reasoning graph for enhanced analysis.
    """
    prompt = "You are a story consistency validator. Detect any **role completeness violations** in the story. Summarize violations per chapter in human-readable paragraph form. If there are no violations, respond 'No Violations. Wohoo!'\n\n"
    
    for chap_id, events in chapters.items():
        prompt += f"Chapter {chap_id}:\n" + "\n".join(events) + "\n\n"
    
    # Add reasoning graph if available
    if reasoning_graph:
        prompt += format_reasoning_graph(reasoning_graph)
    
    return prompt

def call_reasoning_llm(prompt: str, use_reasoning: bool = False) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    system_content = (
        "You are a macro-level story consistency validator.\n"
        "Detect which missing characters/actors, tools, or roles exist in the story.\n"
        "Some actors could be locations.\n"
        "Summarize issues per chapter in human-readable paragraphs.\n"
        "For each violation, guide the user by explicitly mentioning the particular sentence/s you found the violation in.\n"
        "Do NOT reference event IDs or sentence IDs.\n"
        "Do NOT rewrite the story, only report violations."
    )
    
    # Enhanced system message when using reasoning graph
    if use_reasoning:
        system_content += (
            "\n\nYou have been provided with:\n"
            "- Temporal relationships using Allen's Interval Algebra (BEFORE, AFTER, OVERLAPS, etc.)\n"
            "- Causal relationships (CAUSES, ENABLES)\n\n"
            "Use these to detect:\n"
            "1. Timeline contradictions (events out of logical temporal order)\n"
            "2. Causal impossibilities (effects occurring before causes)\n"
            "3. Location/character tracking errors (impossible transitions)\n"
            "4. Narrative coherence breaks"
        )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"[ERROR] LLM request failed: {e}"

# =========================
# Main Function
# =========================

def generate_feedback(csv_path: str, output_dir: str = "output"):
    """Generate feedback with optional reasoning graph integration."""
    chapters = read_csv_as_chapter_text(csv_path)
    
    # Load reasoning graph if available
    reasoning_graph = load_reasoning_graph(output_dir)
    
    # Build prompt with reasoning graph
    prompt = build_prompt(chapters, reasoning_graph)
    
    log("Let me check your story...")
    if reasoning_graph:
        log("Using temporal and causal relationship data for enhanced analysis...")
    
    return call_reasoning_llm(prompt, use_reasoning=bool(reasoning_graph))

# =========================
# CLI Execution
# =========================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python character.py <role_completeness.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output = generate_feedback(csv_path)

    print("\n=== Role Completeness Feedback ===\n")
    print(output)
