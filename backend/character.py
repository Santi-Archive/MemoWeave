# character.py

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

def read_story_text(story_path: str) -> str:
    """
    Reads the raw story text file and returns its full content.
    This is needed so the LLM can analyze narrative context, descriptive
    introductions, and character motivations that are lost in CSV extraction.
    """
    if not os.path.exists(story_path):
        return ""
    
    with open(story_path, "r", encoding="utf-8") as f:
        return f.read()

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

def build_prompt(chapters: Dict[str, List[str]], story_text: str = "") -> str:
    """
    Builds a single macro-level prompt for the LLM using chapter aggregation
    and the full raw story text for contextual analysis.
    """
    prompt = (
        "You are a story consistency validator specializing in **role completeness violations**.\n"
        "Below you will find:\n"
        "1. The FULL RAW STORY TEXT — use this to analyze descriptive introductions, narrative context, and character motivations.\n"
        "2. EXTRACTED ROLE EVENTS per chapter (Actors, Targets, Locations) — use this as a structured guide to the entities present.\n\n"
        "Analyze carefully and report all role completeness violations.\n\n"
    )

    # Include the full story text
    if story_text:
        prompt += "=== FULL RAW STORY TEXT ===\n"
        prompt += story_text + "\n\n"
    
    prompt += "=== EXTRACTED ROLE EVENTS BY CHAPTER ===\n\n"
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
                    "You are an expert narrative consistency validator for fiction.\n"
                    "Your task is to detect **Role Completeness Violations** — places where the story fails to properly introduce, utilize, or maintain logically consistent characters, objects, locations, or motivations.\n\n"
                    
                    "## VIOLATION CATEGORIES TO CHECK\n\n"
                    
                    "1. **Geographic/Directional Contradictions**: When physical directions or geographic logic contradicts itself. "
                    "For example, if a river flows north-to-south, stating the sun rises 'over the western bank' is logically incorrect. Or stating a character crossed a bridge when they were already on the destination side.\n\n"
                    
                    "2. **Orphaned Entities (Introduced but Never Used)**: When a specific person, faction, or object is introduced prominently but never appears or factors into the narrative again. "
                    "For example, if a 'rebel faction' is introduced in Chapter 1 as a major threat but is never mentioned again in the entire story.\n\n"
                    
                    "3. **Unintroduced Entities (Used without Prior Mention)**: When a character, place, or critical object suddenly appears and acts without proper narrative introduction. "
                    "For example, if a scene is set in 'London' but the text suddenly says the character walks the streets of 'Paris' without an intervening travel scene. Or a character named Elias speaking when he was never introduced.\n\n"
                    
                    "4. **Object/Action Semantic Mismatch**: When an object is described performing an action impossible for its nature. "
                    "For example, describing tranquilizer darts producing the sound 'of gunshots' (darts fire quietly), or a sword being 'reloaded'.\n\n"
                    
                    "5. **Location/Setting Inconsistencies**: When a character's immediate location shifts within the same continuous scene without a transition. "
                    "For example, if the opening paragraph places a character 'aboard a train', but the next sentence says they 'arrived at the airport terminal' in the same present moment.\n\n"
                    
                    "6. **Logic Failures in Routine/Established Facts**: When the narrative states an established reality, but characters react in ways that contradict it. "
                    "For example, if a boss has visited an office 'every Tuesday for five years', stating that his arrival on a Tuesday was 'unbeknownst to the staff' is illogical — they should expect him.\n\n"
                    
                    "7. **Dangling/Ambiguous References**: When actions or reactions occur without a clear source. "
                    "For example, writing 'his arrival was greeted with unease' without specifying *who* was uneasy. Or stating 'the realization became clear' inside an empty room with no characters present to experience the realization.\n\n"
                    
                    "8. **Unjustified Action/Motivation (Out of Character)**: When characters behave contrary to their established role, profession, or prior emotional state without explanation. "
                    "For example, a ruthless interrogator suddenly showing deep empathy for a suspect without any triggering event, or a billionaire CEO personally cleaning the company lobby.\n\n"
                    
                    "9. **Unresolved Foreshadowing or Questions**: When a character explicitly notes a mystery, suspicion, or question, but the narrative completely drops the thread. "
                    "For example, 'She wondered why she was chosen for the mission' — but the story never actually answers or explores the motivation.\n\n"
                    
                    "10. **Terminology/Item Inconsistency**: When the narrative switches terms for a specific functional object in ways that change its nature. "
                    "For example, referring to a device initially as a 'listening device' (microphone) and later referring to the exact same device as a 'tracker chip' (GPS). These are distinct tools.\n\n"
                    
                    "## MANDATORY ANALYSIS PROCEDURE\n\n"
                    "You MUST follow these steps in order:\n\n"
                    "Step 1: Read the FULL story text carefully. Identify all named characters, locations, and key objects.\n"
                    "Step 2: Map out the character roles and motivations established early in the story.\n"
                    "Step 3: For EACH chapter, independently apply ALL 10 violation categories above. Check every geographic detail, every object usage, every emotional shift, and every location change.\n"
                    "Step 4: Cross-reference entities ACROSS chapters — check if entities introduced early are abandoned, or if unresolved questions are left hanging at the end of the text.\n\n"
                    "CRITICAL: You MUST analyze EVERY chapter with EQUAL thoroughness. Do NOT focus only on the first chapter. "
                    "Chapters later in the story often contain subtle violations (such as dropped plot threads or unresolved questions) that are easy to miss if you rush through them.\n\n"
                    
                    "## OUTPUT FORMAT\n\n"
                    "Write your findings in sentence and paragraph form, grouped by chapter. Do NOT use tables or columns.\n\n"
                    "For each violation, write a paragraph that:\n"
                    "- Starts with the tag [Role Completeness]\n"
                    "- Quotes the specific sentence(s) or phrase(s) where the violation occurs\n"
                    "- Explains WHY this is a role completeness contradiction, referencing which category it falls under\n"
                    "- References the contradicting passage or established logic it conflicts with\n\n"
                    "If a chapter has no violations, state that briefly. But do NOT declare a chapter violation-free without first checking all 10 categories against every paragraph in that chapter.\n\n"
                    
                    "## RULES\n\n"
                    "- Read the FULL story text carefully before analyzing. Understand character motives and spatial logic.\n"
                    "- Do NOT flag minor descriptive details that do not break narrative understanding.\n"
                    "- Track objects and factions. If something important is introduced, it must serve a purpose.\n"
                    "- Be highly literal regarding physics, geography, and weapon mechanics.\n"
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
    """Generate feedback for role completeness validation."""
    chapters = read_csv_as_chapter_text(csv_path)
    
    # Read raw story text if path is provided
    story_text = ""
    if story_path:
        story_text = read_story_text(story_path)
    
    prompt = build_prompt(chapters, story_text)
    
    log("Let me check your story...")
    log(f"[DATA] Chapters loaded: {list(chapters.keys())}")
    
    return call_reasoning_llm(prompt)

# =========================
# CLI Execution
# =========================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python character.py <role_completeness.csv> [story.txt]")
        sys.exit(1)

    csv_path = sys.argv[1]
    story_path = sys.argv[2] if len(sys.argv) > 2 else None
    output = generate_feedback(csv_path, story_path=story_path)

    print("\n=== Role Completeness Feedback ===\n")
    print(output)
