# LLM Feedback Prompts - Fine-tuning Guide

This document explains where to find and modify the LLM prompts used for generating feedback in MemoWeave.

## Overview

MemoWeave uses LLM (Large Language Model) feedback to detect story consistency violations. There are two types of feedback, each with its own prompt configuration:

1. **Temporal Consistency** - Detects timing contradictions, verb tense mismatches, deictic marker misuse, spatial-temporal contradictions, and event ordering issues
2. **Role Completeness** - Detects missing characters, actors, or tools in events

Both use OpenRouter API with the `gpt-oss-120b` model by default.

## Prompt Locations

### 1. Temporal Consistency Feedback

**File:** [`backend/events.py`](file:///c:/Users/wobin/Documents/GitHub/MemoWeave/backend/events.py)

**Function:** `call_reasoning_llm()` — Contains the system prompt with 9 violation categories

**System Prompt Violation Categories:**
1. Contradictory time signals for the same event (definite vs indefinite)
2. Spatial-temporal contradictions (character in two places at the "same present")
3. Verb tense violations in flashback/present transitions
4. Deictic marker misuse ("Now", "the present") contradicting established timeline
5. Contradictory "present" anchors across paragraphs
6. Adverb-timeframe incompatibility ("Meanwhile" + past durations)
7. Duration contradictions for the same information
8. Time-of-day semantic precision (evening vs night)
9. Cross-chapter timeline continuity

**User Prompt Builder:** `build_prompt()` 
- Includes the **full raw story text** for verb tense and deictic marker analysis
- Aggregates extracted events by chapter from the temporal_consistency.csv
- Formats: `"Chapter {id}: - {event_text} (time: {time}, type: {type})"`

**To Fine-tune:**
- Modify the violation categories in the system prompt to change what the LLM looks for
- Adjust the user prompt format in `build_prompt()` to change data presentation
- Change `MODEL_NAME` (line 17) to use a different model
- Adjust `temperature` for more/less creative responses (currently 0.0)


---

### 2. Role Completeness Feedback

**File:** [`backend/character.py`](file:///c:/Users/wobin/Documents/GitHub/MemoWeave/backend/character.py)

**Function:** `call_reasoning_llm()` — Contains the system prompt with 10 violation categories

**System Prompt Violation Categories:**
1. Geographic/Directional Contradictions
2. Orphaned Entities (Introduced but Never Used)
3. Unintroduced Entities (Used without Prior Mention)
4. Object/Action Semantic Mismatch
5. Location/Setting Inconsistencies
6. Logic Failures in Routine/Established Facts
7. Dangling/Ambiguous References
8. Unjustified Action/Motivation (Out of Character)
9. Unresolved Foreshadowing or Questions
10. Terminology/Item Inconsistency

**User Prompt Builder:** `build_prompt()` 
- Includes the **full raw story text** for analyzing narrative context, descriptive introductions, and character motivations
- Aggregates extracted role events by chapter from the role_completeness.csv
- Formats: `"Chapter {id}: - {event_text} (actor: {actor}, target: {target}, location: {location})"`

**To Fine-tune:**
- Modify the violation categories in the system prompt to change what the LLM looks for
- Adjust the user prompt format in `build_prompt()` to change data presentation
- Change `MODEL_NAME` (line 17) to use a different model
- Adjust `temperature` for more/less creative responses (currently 0.0)

---

## Configuration

Both files share the same configuration structure:

```python
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b:free"
```

**Environment Variable Required:**
- Set `OPENROUTER_API_KEY` in your `.env` file to enable LLM feedback

## Prompt Engineering Tips

When fine-tuning the prompts:

1. **System Prompt Guidelines:**
   - Keep it concise and focused on the specific task
   - Define clear boundaries (what to do and what NOT to do)
   - Use imperative language ("Detect...", "Summarize...", "Do NOT...")

2. **User Prompt Construction:**
   - Organize data hierarchically (by chapter in this case)
   - Include relevant metadata for context
   - Balance verbosity vs. context (too much data can confuse the LLM)

3. **Temperature Settings:**
   - `0.0` = Deterministic, consistent responses (current setting)
   - `0.3-0.7` = More varied but still focused
   - `0.8-1.0` = Creative and diverse (may be inconsistent)

4. **Model Selection:**
   - Current: `gpt-oss-120b:free` (free tier, good balance)
   - For better quality: Consider paid models like `anthropic/claude-3.5-sonnet`
   - Check OpenRouter docs for available models

## Testing Changes

After modifying prompts:

1. Run an analysis through the UI
2. Check the "Memo Weave Feedback" section for output quality
3. Iterate on the prompts based on the feedback quality
4. Consider testing with multiple story samples to ensure consistency

## Example: Making the Feedback More Detailed

**Before (events.py, line 73):**
```python
"Detect temporal contradictions or overlapping events.\n"
```

**After:**
```python
"Detect temporal contradictions or overlapping events.\n"
"For each violation, explain WHY it's inconsistent and suggest what to check.\n"
```

This would make the LLM provide more explanatory feedback with suggestions.
