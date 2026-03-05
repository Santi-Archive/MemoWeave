# events.py

import os
import csv
import requests
from dotenv import load_dotenv
from typing import Callable, List, Dict

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

def build_prompt(chapters: Dict[str, List[str]]) -> str:
    """
    Builds a single macro-level prompt for the LLM using chapter aggregation.
    """
    prompt = "You are a story consistency validator. Detect any **temporal consistency violations** in the story. Summarize violations per chapter in human-readable paragraph form.\n\n"
    
    for chap_id, events in chapters.items():
        prompt += f"Chapter {chap_id}:\n" + "\n".join(events) + "\n\n"
    
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
                    "You are a macro-level story consistency validator.\n"
                    "Detect temporal inconsistencies such as contradictory timelines, impossible event order, or overlapping actions that cannot logically occur.\n"
                    "Know the whole context first, especially in case of flashbacks, memories, or time skips.\n"
                    "For each sentence, carefully check whether the timing and order of events are consistent with the surrounding context.\n"
                    "Flag violations if events occur before their stated cause, if a character appears in two places at the same time, or if actions overlap in ways that are physically impossible.\n"
                    "Also flag cases where the narrative states that an event happened earlier or later but contradicts previously established timing.\n"
                    "Flag violations only if the temporal sequence is clearly contradictory, impossible, or inconsistent with previously established events.\n"
                    "Summarize issues per chapter in human-readable paragraphs.\n"
                    "For each violation, guide the user by explicitly mentioning the particular sentence/s you found the violation in.\n"
                    "Do NOT flag minor pacing issues, implied time gaps, or normal flashbacks that do not contradict the timeline.\n"
                    "Do NOT reference event IDs or sentence IDs.\n"
                    "Do NOT rewrite the story, only report violations.\n"
                    "Be precise, analytical, and avoid speculative interpretations while still identifying clear temporal conflicts to maximize agreement with a human annotator."
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
    prompt = build_prompt(chapters)

    log("Sending to LLM API...")
    log(f"[DATA] Chapters loaded: {list(chapters.keys())}")
    log(f"[PROMPT]\n{prompt}")

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
