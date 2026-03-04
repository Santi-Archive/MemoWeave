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

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are operating in Semi-Sensitive Audit Mode (60% strictness) for Role Completeness and Object Continuity.\n"
                    "Your objective is to evaluate the story sentence-by-sentence to detect missing roles, unintroduced entities, and continuity errors.\n"
                    "Your task:\n"
                    "For EACH sentence in the chapter, classify it as either 'Violation' or 'Valid'. Classify EVERY sentence in the story. No skipping.\n"
                    "RULES FOR CLASSIFYING AS 'VIOLATION' (A sentence is a Violation if ANY of the following occur):\n"
                    "1. Missing Elements: An action occurs without a clearly defined actor, a required target is missing, a required location is undefined, or a tool is used without being logically acquired beforehand.\n"
                    "2. Entity & Continuity Errors: A character or object appears without prior introduction, a removed entity reappears without explanation, or an object disappears or changes state inexplicably.\n"
                    "3. Role & Capability Errors: A character performs an action beyond their previously established ability, the responsible actor is ambiguous, or a required role is implicitly assumed rather than stated.\n"
                    "STRICT RULES & INFERENCE GUIDELINES:\n"
                    "- If even ONE issue applies → classify as 'Violation'.\n"
                    "- If uncertain → classify as 'Violation', but do NOT over-classify. You are permitted to make logical inferences based on past context to maximize agreement with human annotators.\n"
                    "- MINIMALLY give the text the benefit of the doubt. Do NOT assume omitted information is automatically acceptable.\n"
                    "- Do NOT rewrite, summarize, or merge sentences.\n"
                    "Output EXACTLY in this format for every single sentence to maximize token efficiency:\n"
                    "<sentence_1_snippet>: Violation | <short_explanation>\n"
                    "<sentence_2_snippet>: Valid | <short_explanation>\n"
                    "<sentence_3_snippet>: Violation | <short_explanation>\n"
                    "Explanations must be average-length and human-friendly. Provide NO long commentary."
                )
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