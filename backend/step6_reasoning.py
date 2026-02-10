
import os
import json
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Use relative imports if running as module, otherwise add parent to path
try:
    from .utils import load_json, save_json, ensure_directory
    from .llm_client import call_gemini_with_retry as call_reasoning_model
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backend.utils import load_json, save_json, ensure_directory
    from backend.llm_client import call_gemini_with_retry as call_reasoning_model

def format_events_for_prompt(events: List[Dict]) -> str:
    """
    Format events into the JSON structure expected by the prompt.
    Minimizing fields to what's relevant for reasoning.
    """
    simplified_events = []
    for ev in events:
        # Extract minimal needed info
        sim = {
            "event_id": ev.get("event_id"),
            "predicate": ev.get("text", "") or ev.get("lemma", ""),
            "agent_entity_id": ev.get("actor", "UNKNOWN"),
            "patient_entity_id": None, # Extract from args if available
            "time_start": ev.get("time", {}).get("normalized"),
            "time_end": None,
            "sentence_id": ev.get("sentence_id"),
            "document_id": ev.get("chapter_id")
        }
        
        # Try to find patient/object in args
        # This depends on the exact structure of 'args' in events.json which can vary
        # For now, we leave it null or try simple extraction
        if "args" in ev and isinstance(ev["args"], list):
             for arg in ev["args"]:
                 if arg.get("role") in ["ARG1", "PATIENT", "THEME"]:
                     sim["patient_entity_id"] = arg.get("text")
                     break

        simplified_events.append(sim)
        
    return json.dumps(simplified_events, indent=2)

def format_entities_for_prompt(entities: Dict) -> str:
    """
    Format entities for the prompt.
    Using the canonical entities from memory module.
    """
    # entities input is expected to be the 'entities' key from memory_module
    # which has 'by_label', 'entities', etc.
    
    # We want a simple list of normalized entities
    # Let's extract from 'by_label'
    
    normalized_list = []
    
    if "by_label" in entities:
        for label, items in entities["by_label"].items():
            for item in items:
                normalized_list.append(f"{item['text']} ({label})")
    
    return json.dumps(normalized_list, indent=2)


def generate_reasoning_prompt(events: List[Dict], entities: Dict) -> Dict[str, str]:
    """
    Construct the System and User prompts based on the template.
    """
    
    system_prompt = """
You are a deterministic temporal–causal reasoning module embedded inside an automated NLP pipeline.

All upstream processing has already been completed. 
You will be given:
- normalized entities
- resolved coreference
- extracted event frames
- normalized temporal intervals
- sentence and document provenance

Your responsibilities are strictly limited to:
1. Inferring temporal relations between events using Allen’s Interval Algebra.
2. Inferring causal or enabling relations between events following Pearl-style causal graph principles.
3. Producing graph-structured outputs suitable for direct ingestion into Neo4j.
4. Preserving pipeline determinism and schema consistency.

You must assume:
- All provided events are valid.
- All timestamps are normalized.
- No additional extraction is required.

You must NOT:
- Modify, merge, or invent events or entities.
- Output natural language explanations.
- Perform embedding computation.
- Violate temporal precedence when asserting causality.

Your output must conform exactly to the declared schema.
"""

    user_prompt_template = """
### PIPELINE CONTEXT
This request is part of an automated execution of pipeline.py.
Upstream modules have already completed preprocessing, extraction, and normalization.

### ENTITIES (NORMALIZED)
{entities_json}

### EVENTS (FINAL EVENT FRAMES)
Each event includes:
- event_id
- predicate/action
- agent_entity_id
- patient_entity_id (nullable)
- time_start
- time_end
- sentence_id
- document_id

{events_json}

### EXISTING RELATIONS
If any relations already exist, they are listed here.
If none exist, this section will be empty.

### CONSTRAINTS
- Temporal relations must be derived using Allen’s Interval Algebra.
- Causal relations must respect Pearl-style directed acyclic causality.
- Causality requires temporal precedence.
- Only infer relations within the same document or narrative scope.

### TASK
1. Infer temporal relations between events.
2. Infer causal or enabling relations where narratively and temporally justified.
3. Output ONLY a valid JSON object with the following structure:
{{
  "temporal_relations": [
    {{"from_event": "event_id_A", "to_event": "event_id_B", "relation": "BEFORE|AFTER|OVERLAPS|..."}}
  ],
  "causal_relations": [
    {{"from_event": "event_id_A", "to_event": "event_id_B", "relation": "CAUSES|ENABLES"}}
  ]
}}
4. Do NOT output markdown code blocks (```json ... ```). Just the raw JSON string.
"""

    user_prompt = user_prompt_template.format(
        entities_json=format_entities_for_prompt(entities),
        events_json=format_events_for_prompt(events)
    )
    
    return {
        "system": system_prompt,
        "user": user_prompt
    }


def export_to_cypher(events: List[Dict], reasoning_output: Dict[str, Any], output_path: Path) -> None:
    """
    Generate Neo4j Cypher import script from events and reasoning output.
    """
    lines = []
    lines.append("// Info: Generated by Temporal-Memory Pipeline Step 6")
    lines.append(f"// Date: {datetime.now().isoformat()}")
    lines.append("")
    
    lines.append("// 1. Create Constraints")
    lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;")
    lines.append("")
    
    lines.append("// 2. Create Event Nodes")
    for ev in events:
        eid = ev.get("event_id")
        text = ev.get("text", "").replace("'", "\\'")
        lemma = ev.get("lemma", "").replace("'", "\\'")
        lines.append(f"MERGE (e:Event {{id: '{eid}'}}) SET e.text = '{text}', e.lemma = '{lemma}';")
    
    lines.append("")
    lines.append("// 3. Create Temporal Relations")
    temp_rels = reasoning_output.get("temporal_relations", [])
    for rel in temp_rels:
        src = rel.get("from_event")
        dst = rel.get("to_event")
        rel_type = rel.get("relation", "BEFORE").upper() # Usually BEFORE, OVERLAPS, etc.
        
        if src and dst:
            lines.append(f"MATCH (a:Event {{id: '{src}'}}), (b:Event {{id: '{dst}'}})")
            lines.append(f"MERGE (a)-[:{rel_type}]->(b);")

    lines.append("")
    lines.append("// 4. Create Causal Relations")
    causal_rels = reasoning_output.get("causal_relations", [])
    for rel in causal_rels:
        src = rel.get("from_event")
        dst = rel.get("to_event")
        rel_type = rel.get("relation", "CAUSES").upper() # CAUSES, ENABLES, etc.
        
        if src and dst:
            lines.append(f"MATCH (a:Event {{id: '{src}'}}), (b:Event {{id: '{dst}'}})")
            lines.append(f"MERGE (a)-[:{rel_type}]->(b);")
            
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Neo4j import script saved to {output_path}")

def process_reasoning(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Main function for Step 6.
    1. Load memory_module (or events/entities raw).
    2. Generate Prompt.
    3. Call LLM.
    4. Save output.
    """
    print(f"Loading data for reasoning from {input_dir}...")
    
    # We can load the full memory module if step 5 ran, or individual files
    memory_path = Path(output_dir) / "memory" / "memory_module.json"
    
    if memory_path.exists():
        memory_data = load_json(str(memory_path))
        events = memory_data.get("events", [])
        entities = memory_data.get("entities", {})
    else:
        # Fallback to individual files
        print("Memory module not found, loading individual files...")
        events_data = load_json(f"{input_dir}/memory/events.json")
        events = events_data.get("events", [])
        entities = {} 

    print("Generating prompts...")
    prompts = generate_reasoning_prompt(events, entities)
    
    # Save prompt for debugging
    debug_prompt_path = Path(output_dir) / "debug_reasoning_prompt.txt"
    with open(debug_prompt_path, "w", encoding="utf-8") as f:
        f.write("=== SYSTEM ===\n")
        f.write(prompts["system"])
        f.write("\n\n=== USER ===\n")
        f.write(prompts["user"])
    #print(f"Debug prompt saved to {debug_prompt_path}")

    # Call LLM
    print("Calling Reasoning Model...")
    try:
        response_str = call_reasoning_model(prompts["system"], prompts["user"])
    except Exception as e:
        print(f"Error calling LLM: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

    # Parse Response
    try:
        clean_response = response_str.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        
        reasoning_output = json.loads(clean_response)
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM JSON response: {e}")
        print(f"Raw response: {response_str[:500]}...")
        return {"status": "failed", "error": "json_parse_error", "raw_response": response_str}

    # Save JSON
    output_path = Path(output_dir) / "memory" / "reasoning_graph.json"
    ensure_directory(str(output_path.parent))
    save_json(reasoning_output, str(output_path))
    print(f"Reasoning graph saved to {output_path}")

    # Export to Neo4j Cypher
    cypher_path = Path(output_dir) / "memory" / "import_neo4j.cypher"
    try:
        export_to_cypher(events, reasoning_output, cypher_path)
    except Exception as e:
        print(f"Error exporting to Cypher: {e}")

    return {
        "status": "success",
        "output_path": str(output_path),
        "data": reasoning_output
    }

if __name__ == "__main__":
    # Test run
    import sys
    if len(sys.argv) > 1:
        process_reasoning(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])