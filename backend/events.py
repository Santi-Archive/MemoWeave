# events.py

import os
import csv
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

def read_story_text(story_path: str) -> str:
    """
    Reads the raw story text file and returns its full content.
    This is needed so the LLM can analyze verb tenses, deictic markers,
    and exact phrasings that are lost in CSV extraction.
    """
    if not os.path.exists(story_path):
        return ""
    
    with open(story_path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(chapters: Dict[str, List[str]], story_text: str = "") -> str:
    """
    Builds a single macro-level prompt for the LLM using chapter aggregation
    and the full raw story text for contextual analysis.
    """
    prompt = (
        "You are a story consistency validator specializing in **temporal consistency violations**.\n"
        "Below you will find:\n"
        "1. The FULL RAW STORY TEXT — use this to analyze verb tenses, deictic markers, exact phrasings, and spatial-temporal context.\n"
        "2. EXTRACTED TEMPORAL EVENTS per chapter — use this as a structured guide to the temporal signals present.\n\n"
        "Analyze carefully and report all temporal consistency violations.\n\n"
    )

    # Include the full story text
    if story_text:
        prompt += "=== FULL RAW STORY TEXT ===\n"
        prompt += story_text + "\n\n"
    
    prompt += "=== EXTRACTED TEMPORAL EVENTS BY CHAPTER ===\n\n"
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
                    "You are an expert temporal consistency validator for narrative fiction.\n"
                    "Your task is to detect **temporal consistency violations** — places where the story's use of time signals, verb tenses, or temporal framing is contradictory, inconsistent, or logically impossible.\n\n"
                    
                    "## VIOLATION CATEGORIES TO CHECK\n\n"
                    
                    "1. **Contradictory Time Signals for the Same Event**: When two different temporal references point to the same event but contradict each other. "
                    "For example, if a debriefing is referred to as both 'weeks earlier' and 'Three months prior' within connected passages, "
                    "the indefinite 'weeks' contradicts the definite 'three months'. Flag when the same event is anchored to incompatible time references.\n\n"
                    
                    "2. **Spatial-Temporal Contradictions**: When the narrative establishes a character's present location and time-of-day in one paragraph, "
                    "but a later paragraph in the same 'present' context places them at a different location with a contradictory time-of-day. "
                    "For example, if the opening establishes 'early morning sun' on a yacht in Palawan as the present, but later the 'present day' shows 'mid-day heat of Mactan', "
                    "the character cannot be in both places at the same present time.\n\n"
                    
                    "3. **Verb Tense Violations in Flashback/Present Transitions**: When a passage is explicitly framed as a flashback (e.g., 'Back to three months earlier') "
                    "but uses present-tense verbs (e.g., 'sits', 'lingers sharply') instead of past tense. "
                    "Also flag when a present-tense verb like 'lingers' is used to describe a memory right before a sentence that transitions into a flashback.\n\n"
                    
                    "4. **Deictic Marker Misuse ('Now', 'the present')**: The word 'Now' anchors the reader to the story's established present timeline. "
                    "If 'Now' appears inside a passage that is clearly a flashback or recovery period, it contradicts the established present. "
                    "For example, if the present is set on a yacht in Palawan, using 'Now' in a passage about reading a dossier during recovery (which is a flashback) is a violation.\n\n"
                    
                    "5. **Contradictory 'Present' Anchors**: When the narrative establishes one scene as 'the present' (e.g., opening paragraph on a yacht), "
                    "but later uses 'back to the present' to introduce a completely different scene with contradictory details (different location, different time of day). "
                    "This cancels out the original established present.\n\n"
                    
                    "6. **Adverb-Timeframe Incompatibility**: When temporal adverbs contradict the time range they introduce. "
                    "For example, 'Meanwhile' implies simultaneous action in the present, but 'in the past several weeks' describes an extended past duration. "
                    "These are incompatible when combined in the same sentence. Also check that verb tenses match the intended timeframe.\n\n"
                    
                    "7. **Duration Contradictions for the Same Information**: When the same piece of intelligence or information is described with contradictory time ranges. "
                    "For example, if intel was 'compiled over the past year' but the same information is said to have 'surfaced from former informants months earlier', "
                    "these durations don't align for the same data.\n\n"
                    
                    "8. **Time-of-Day Semantic Precision**: Check that time-of-day words are used with correct semantic meaning. "
                    "'Night' (roughly 9 PM to 4 AM, associated with darkness and sleep) is distinct from 'evening' (roughly 5 PM to 9 PM, associated with twilight and dinner). "
                    "If a scene uses 'night' but contextual details (arriving under 'dim glow of early evening', dinner activities) indicate evening, flag the inconsistency.\n\n"
                    
                    "9. **Cross-Chapter Timeline Continuity**: Track when narrative timelines begin AND end across chapters. "
                    "If a present-day timeline in Mactan is concluded in a specific chapter (e.g., the scene ends with a revelation in a server room), "
                    "then a later reference to carrying something 'into the present timeline of Mactan' is a violation because that timeline has already ended.\n\n"
                    
                    "## MANDATORY ANALYSIS PROCEDURE\n\n"
                    "You MUST follow these steps in order:\n\n"
                    "Step 1: Read the FULL story text carefully. Identify the story's 'anchor present' (usually established in the opening paragraph).\n"
                    "Step 2: Map out the narrative timeline layer for EACH chapter — identify which paragraphs are present, which are flashbacks, and where transitions occur.\n"
                    "Step 3: For EACH chapter, independently apply ALL 9 violation categories above. Check every temporal signal, every verb tense, every deictic marker, every time-of-day word.\n"
                    "Step 4: Cross-reference timelines ACROSS chapters — check if later chapters reference timelines that have already concluded in earlier chapters.\n\n"
                    "CRITICAL: You MUST analyze EVERY chapter with EQUAL thoroughness. Do NOT focus only on the first chapter. "
                    "Chapters later in the story often contain subtle violations (such as time-of-day precision errors or cross-chapter timeline references) that are easy to miss if you rush through them.\n\n"
                    
                    "## OUTPUT FORMAT\n\n"
                    "Write your findings in sentence and paragraph form, grouped by chapter. Do NOT use tables or columns.\n\n"
                    "For each violation, write a paragraph that:\n"
                    "- Starts with the tag [Temporal Consistency]\n"
                    "- Quotes the specific sentence(s) or phrase(s) where the violation occurs\n"
                    "- Explains WHY this is a temporal contradiction\n"
                    "- References the contradicting passage or established timeline it conflicts with\n\n"
                    "If a chapter has no violations, state that briefly. But do NOT declare a chapter violation-free without first checking all 9 categories against every paragraph in that chapter.\n\n"
                    
                    "## RULES\n\n"
                    "- Read the FULL story text carefully before analyzing. Understand the narrative structure including all flashbacks, present timelines, and time shifts.\n"
                    "- Identify which scene is the story's 'anchor present' (usually established in the opening paragraph).\n"
                    "- Track every transition between present and flashback, noting how each is framed.\n"
                    "- Pay close attention to verb tenses — they must be consistent with whether a passage is present or flashback.\n"
                    "- Pay close attention to deictic markers ('Now', 'the present', 'back to') — they must be consistent with the established timeline layers.\n"
                    "- Pay close attention to time-of-day words — 'night' and 'evening' are NOT interchangeable. Check contextual clues.\n"
                    "- Track which present-day timelines have concluded. A timeline ends when the narrative moves permanently away from it. Later references to that timeline as ongoing are violations.\n"
                    "- Do NOT flag normal flashback transitions that are properly framed and internally consistent.\n"
                    "- Do NOT reference event IDs or sentence IDs.\n"
                    "- Do NOT rewrite the story, only report violations.\n"
                    "- Be precise and analytical. Quote the exact text where violations occur.\n"
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

def generate_feedback(csv_path: str, output_dir: str = "output", story_path: str = None):
    chapters = read_csv_as_chapter_text(csv_path)
    
    # Read raw story text if path is provided
    story_text = ""
    if story_path:
        story_text = read_story_text(story_path)
    
    prompt = build_prompt(chapters, story_text)

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
        print("Usage: python events.py <temporal_consistency.csv> [story.txt]")
        sys.exit(1)

    csv_path = sys.argv[1]
    story_path = sys.argv[2] if len(sys.argv) > 2 else None
    output = generate_feedback(csv_path, story_path=story_path)

    print("\n=== Temporal Consistency Feedback ===\n")
    print(output)
