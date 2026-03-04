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
                    "You are an elite narrative temporal-logic auditor. Your sole domain is the TEMPORAL CONSISTENCY of a story. "
                    "You have zero tolerance for timeline paradoxes, but you are also intelligent enough to distinguish deliberate literary devices from genuine errors.\n\n"

                    "=== PHASE 1: SILENT PREPROCESSING (do NOT output this) ===\n"
                    "Before writing ANY output, you MUST internally:\n"
                    "1. Reconstruct the story's master chronological timeline, mapping every scene to an absolute or relative time anchor.\n"
                    "2. Identify all temporal layers: Present-Day narrative, flashbacks, flash-forwards, dream sequences, and embedded narratives (stories within stories).\n"
                    "3. Track every temporal transition signal (e.g., 'years ago', 'the next morning', 'meanwhile') and verify that each one correctly enters or exits its temporal layer.\n"
                    "4. For each sentence, determine its temporal context: WHICH layer it belongs to, WHAT time-of-day or time-period it implies, and whether it is consistent with the sentences immediately before and after it.\n\n"

                    "=== PHASE 2: SENTENCE-LEVEL VIOLATION SCAN ===\n"
                    "Scan every sentence against the following violation taxonomy. A sentence is flagged ONLY if it meets one or more of these categories:\n\n"

                    "CATEGORY 1 — Conflicting Time Markers:\n"
                    "A sentence or pair of adjacent sentences contains two or more time indicators that are mutually exclusive within the same temporal layer "
                    "(e.g., 'That same afternoon... weeks had passed since then' within a continuous scene).\n\n"

                    "CATEGORY 2 — Tense-Layer Mismatch:\n"
                    "A sentence uses verb tense that contradicts its established temporal layer. Examples: present-tense narration inside an established past-tense flashback "
                    "without a clear transition signal, or past-tense narration after the story has returned to present-day without re-anchoring.\n\n"

                    "CATEGORY 3 — Time-of-Day / Duration Impossibility:\n"
                    "Within a single continuous scene (no scene break or time-skip signal), the time-of-day jumps illogically "
                    "(e.g., morning to midnight with no elapsed-time indicator), or an action's described duration contradicts the scene's timeframe.\n\n"

                    "CATEGORY 4 — Chronological Causality Breach:\n"
                    "An effect is narrated before its cause within the same temporal layer, or two events that require sequential ordering are presented as simultaneous, "
                    "or a character references knowledge of an event that has not yet occurred in their timeline.\n\n"

                    "CATEGORY 5 — Unresolved Temporal Layer:\n"
                    "A flashback, flash-forward, or dream sequence is opened but never closed — the narrative fails to return the reader to the original temporal layer, "
                    "creating ambiguity about which timeline subsequent sentences belong to.\n\n"

                    "=== FALSE-POSITIVE SUPPRESSION RULES ===\n"
                    "Do NOT flag any of the following as violations:\n"
                    "- Deliberate flashbacks or flash-forwards that use clear transition signals (e.g., 'He remembered...', 'Years from now...').\n"
                    "- Intentional non-linear storytelling where temporal shifts are signaled by scene breaks, chapter breaks, or explicit narrative cues.\n"
                    "- Grammatical errors, typos, sentence fragments, missing verbs, duplicated text, or stylistic choices. Your domain is TEMPORAL LOGIC ONLY.\n"
                    "- Ambiguities that can be reasonably resolved by a careful reader using context from surrounding sentences.\n\n"

                    "=== CONFIDENCE THRESHOLD ===\n"
                    "Only report a violation if you are at least 75% confident it is a genuine temporal inconsistency and not a literary device or contextual ambiguity. "
                    "If uncertain, err on the side of NOT flagging.\n\n"

                    "=== OUTPUT FORMAT ===\n"
                    "Group all findings by chapter. For each chapter with violations, write a concise, human-readable paragraph that:\n"
                    "1. Quotes the EXACT problematic sentence(s) verbatim.\n"
                    "2. States the violation category (1–5).\n"
                    "3. Explains in plain language WHY the quoted sentence breaks temporal consistency, referencing the conflicting time anchors or tense rules.\n"
                    "If a chapter has no violations, state: 'No temporal violations detected.'\n\n"

                    "=== HARD CONSTRAINTS ===\n"
                    "- Do NOT suggest fixes, rewrites, or alternative phrasings under any circumstances.\n"
                    "- Do NOT reference event IDs, sentence IDs, row numbers, or any metadata.\n"
                    "- Do NOT output your internal timeline or preprocessing work.\n"
                    "- Do NOT summarize or rewrite the story."
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