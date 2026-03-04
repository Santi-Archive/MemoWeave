# events.py

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

def read_csv_as_chapter_text(csv_path: str) -> Dict[str, List[str]]:
    """
    Reads temporal_consistency.csv and aggregates events per chapter
    into semi-narrative strings for LLM.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    chapters = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chapter_id = row.get("chapter_id", "unknown")
            text = f"- {row.get('text', '')} (time: {row.get('time_raw', 'N/A')}, type: {row.get('time_type', 'N/A')})"
            chapters.setdefault(chapter_id, []).append(text)

    return chapters

def build_prompt(chapters: Dict[str, List[str]], reasoning_rows: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Builds a single macro-level prompt for the LLM using chapter aggregation.
    Optionally includes reasoning graph relationships for enhanced analysis.
    """
    prompt = "You are a story consistency validator. Detect any **temporal inconsistencies** in the story. Summarize violations per chapter in human-readable paragraph form. If there are no violations, respond 'No Violations.'\n\n"
    
    for chap_id, events in chapters.items():
        prompt += f"Chapter {chap_id}:\n" + "\n".join(events) + "\n\n"

    if reasoning_rows:
        prompt += "### KNOWN TEMPORAL & CAUSAL RELATIONSHIPS ###\n"
        for row in reasoning_rows:
            prompt += f"{row.get('from_event', '')} {row.get('relation', '')} {row.get('to_event', '')} ({row.get('type', '')})\n"
        prompt += "\n"

    return prompt

def call_reasoning_llm(prompt: str) -> str:
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
                    "You are operating in High-Sensitivity Sentence Audit Mode strictly calibrated for Temporal Logic.\n"
                    "Your objective is to maximize inter-rater reliability by aggressively targeting specific micro-temporal clashes while ignoring valid narrative devices.\n"
                    "Your task:\n"
                    "For EACH sentence in the chapter, classify it as either:\n"
                    "- Violation\n"
                    "- Valid\n"
                    "You must classify EVERY sentence. No skipping.\n"
                    "RULES FOR CLASSIFYING AS 'VIOLATION':\n"
                    "1. Transition Clash: A front-loaded transition word (e.g., Meanwhile, Now, Earlier, Returning to the present) logically contradicts the timeframe or verb tense used later in that exact same sentence.\n"
                    "2. Unanchored Time Jump: The sentence creates an abrupt, illogical shift in the time of day from the preceding sentence without establishing time passage.\n"
                    "RULES FOR CLASSIFYING AS 'VALID' (Do NOT flag):\n"
                    "3. Valid Flashbacks: Standard flashback framing where the past-perfect tense is correctly established (e.g., 'Three months prior... she had been').\n"
                    "4. Narrative Summaries: Sentences that merely summarize how past events led to the present moment (e.g., 'The past few weeks of planning had led to the present').\n"
                    "5. Grammatical Errors: Missing verbs, fragments, or punctuation errors that do not create a literal temporal paradox.\n"
                    "STRICT RULES:\n"
                    "- If Rule 1 or 2 applies → classify as 'Violation'.\n"
                    "- If Rule 3, 4, or 5 applies → classify as 'Valid'.\n"
                    "- If uncertain whether it is a temporal paradox or just bad grammar → classify as 'Valid'.\n"
                    "- Do NOT rewrite sentences.\n"
                    "- Do NOT summarize.\n"
                    "- Do NOT merge sentences.\n"
                    ""
                    "You must output EXACTLY in this format:\n"
                    "Sentence 1: Violation | <short_explanation based on Rule 1 or 2>\n"
                    "Sentence 2: Valid | <short_explanation based on Rule 3, 4, or 5>\n"
                    "Sentence 3: Violation | <short_explanation based on Rule 1 or 2>\n"
                    "Succinct, but human-friendly explanations only.\n"
                    "No long commentary."
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
    chapters = read_csv_as_chapter_text(csv_path)

    reasoning_rows = load_reasoning_csv(output_dir)

    prompt = build_prompt(chapters, reasoning_rows)

    log("Let me check your story...")
    if reasoning_rows:
        log("Using temporal and causal relationship data for enhanced analysis...")

    return call_reasoning_llm(prompt)

# =========================
# CLI Execution
# =========================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python events.py <temporal_consistency.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output = generate_feedback(csv_path)

    print("\n=== Temporal Consistency Feedback ===\n")
    print(output)