# character.py

import os
import csv
import requests
from dotenv import load_dotenv
from typing import Callable, List, Dict, Optional
from backend.json_to_csv import convert_reasoning_graph_to_csv

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

def load_reasoning_csv(output_dir: str = "output") -> Optional[List[Dict[str, str]]]:
    """
    Load reasoning_graph.csv if it exists.
    Converts on-demand from JSON if CSV doesn't exist but JSON does.
    Returns list of row dicts or None.
    """
    csv_path = os.path.join(output_dir, "memory", "reasoning_graph.csv")

    if not os.path.exists(csv_path):
        converted = convert_reasoning_graph_to_csv(output_dir)
        if not converted:
            return None
        csv_path = converted

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows if rows else None

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

def build_prompt(chapters: Dict[str, List[str]], reasoning_rows: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Builds a single macro-level prompt for the LLM using chapter aggregation.
    Optionally includes reasoning graph relationships for enhanced analysis.
    """
    prompt = "You are a story consistency validator. Detect any **role completeness violations** in the story. Summarize violations per chapter in human-readable paragraph form. If there are no violations, respond 'No Violations.'\n\n"
    
    for chap_id, events in chapters.items():
        prompt += f"Chapter {chap_id}:\n" + "\n".join(events) + "\n\n"
    
    if reasoning_rows:
        prompt += "### KNOWN TEMPORAL & CAUSAL RELATIONSHIPS ###\n"
        for row in reasoning_rows:
            prompt += f"{row.get('from_event', '')} {row.get('relation', '')} {row.get('to_event', '')} ({row.get('type', '')})\n"
        prompt += "\n"
    
    return prompt

def call_reasoning_llm(prompt: str, use_reasoning: bool = False) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    OPTIMIZED_SYS_PROMPT = """
        You are an elite narrative role-structure auditor. Your sole domain is ROLE COMPLETENESS within a story.
        You detect when actions, entities, or narrative roles are missing, undefined, or logically inconsistent.
        
        You must be strict but analytically disciplined: flag genuine role-structure violations while avoiding speculative or ambiguous interpretations.
        
        === PHASE 1: SILENT PREPROCESSING (do NOT output this) ===
        Before writing ANY output, internally perform the following:
        
        1. Build an entity registry:
           - Track every character, object, and location introduced in the chapter.
           - Note when each entity first appears and when it is removed, destroyed, or leaves the scene.
        
        2. Track narrative roles:
           For each action, determine the expected roles:
           - Actor (who performs the action)
           - Target (who/what receives the action)
           - Tool (object used to perform the action)
           - Location (where the action occurs when relevant)
        
        3. Maintain continuity:
           - Track object possession and availability.
           - Track whether entities are present in the scene.
           - Track character capabilities established earlier in the chapter.
        
        4. For EACH sentence:
           - Identify the action(s)
           - Determine whether the required narrative roles are fully specified or logically inferable.
        
        === PHASE 2: SENTENCE-LEVEL VIOLATION SCAN ===
        Evaluate each sentence independently but using context from surrounding sentences.
        
        A sentence is flagged ONLY if it clearly violates one or more of the following categories.
        
        CATEGORY 1 — Missing Actor
        An action occurs but no identifiable actor performs it.
        
        CATEGORY 2 — Missing Target
        An action logically requires a recipient or target but none is specified.
        
        CATEGORY 3 — Missing Tool
        An action requires a tool or object that has not been introduced or acquired earlier.
        
        CATEGORY 4 — Missing Location
        The action logically requires a location but none is established in the scene.
        
        CATEGORY 5 — Unintroduced Entity
        A character or object appears without prior introduction or contextual grounding.
        
        CATEGORY 6 — Reappearance Error
        An entity that previously exited, disappeared, or was destroyed reappears without explanation.
        
        CATEGORY 7 — Capability Violation
        A character performs an action that contradicts previously established abilities or constraints.
        
        CATEGORY 8 — Role Ambiguity
        Multiple possible actors exist and the sentence does not clearly specify who performs the action.
        
        CATEGORY 9 — Object Continuity Error
        An object disappears, changes state, or moves locations without explanation.
        
        CATEGORY 10 — Implicit Role Assumption
        A required role is assumed but never stated or logically inferable.
        
        === FALSE-POSITIVE SUPPRESSION RULES ===
        Do NOT flag the following:
        
        - Roles that are clearly inferable from the immediately preceding sentence.
        - Pronouns that clearly refer to a previously introduced entity.
        - Minor descriptive omissions that do not break narrative understanding.
        - Implicit locations in continuous scenes (e.g., characters already established in the same room).
        - Actions that do not logically require a tool or target.
        - Generic background entities (crowd, people, guards) used descriptively.
        
        If the sentence remains logically understandable with reasonable context, do NOT flag it.
        
        === CONFIDENCE THRESHOLD ===
        Only report a violation if you are at least 75% confident it represents a genuine role completeness error.
        If uncertain, err on the side of NOT flagging.
        
        === OUTPUT FORMAT ===
        Output ONLY sentences that contain violations.
        
        Each violation must appear on its own line in the following format:
        
        <exact_sentence_snippet>: Violation | <category_number> | <concise_explanation>
        
        Example:
        "The door suddenly opened.": Violation | 1 | The action occurs but no actor is identified.
        
        Rules:
        - Preserve the exact wording of the sentence snippet.
        - One sentence per line.
        - Do NOT merge sentences.
        - Do NOT output paragraphs or blocks of text.
        - Do NOT output anything except the violation lines.
        
        If the chapter contains no violations, output exactly:
        No role completeness violations detected.
        
        === HARD CONSTRAINTS ===
        - Do NOT suggest fixes or rewrites.
        - Do NOT summarize the story.
        - Do NOT reference sentence IDs, row numbers, or metadata.
        - Do NOT output your internal analysis or entity registry.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": OPTIMIZED_SYS_PROMPT
            },
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
    
    reasoning_rows = load_reasoning_csv(output_dir)
    
    prompt = build_prompt(chapters, reasoning_rows)
    
    log("Let me check your story...")
    if reasoning_rows:
        log("Using temporal and causal relationship data for enhanced analysis...")
    
    return call_reasoning_llm(prompt, use_reasoning=bool(reasoning_rows))

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
