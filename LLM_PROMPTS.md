# LLM Feedback Prompts - Fine-tuning Guide

This document explains where to find and modify the LLM prompts used for generating feedback in MemoWeave.

## Overview

MemoWeave uses LLM (Large Language Model) feedback to detect story consistency violations. There are two types of feedback, each with its own prompt configuration:

1. **Temporal Consistency** - Detects timing contradictions and event ordering issues
2. **Role Completeness** - Detects missing characters, actors, or tools in events

Both use OpenRouter API with the `gpt-oss-120b:free` model by default.

## Prompt Locations

### 1. Temporal Consistency Feedback

**File:** [`backend/events.py`](file:///c:/Users/Gilbert/MemoWeave/backend/events.py)

**Function:** `call_reasoning_llm()` (lines 59-90)

**System Prompt** (lines 70-77):
```python
"You are a macro-level story consistency validator.\n"
"Know the whole context first, in case of flashback sequences"
"Detect temporal contradictions or overlapping events.\n"
"Summarize issues per chapter in human-readable paragraphs.\n"
"Do NOT reference event IDs or sentence IDs.\n"
"Do NOT rewrite the story, only report violations."
```

**User Prompt Builder:** `build_prompt()` (lines 48-57)
- Aggregates events by chapter from the temporal_consistency.csv
- Formats: `"Chapter {id}: - {event_text} (time: {time}, type: {type})"`

**To Fine-tune:**
- Modify the system prompt content in lines 71-76 to change LLM behavior
- Adjust the user prompt format in `build_prompt()` to change data presentation
- Change `MODEL_NAME` (line 17) to use a different model
- Adjust `temperature` (line 81) for more/less creative responses (currently 0.0)

---

### 2. Role Completeness Feedback

**File:** [`backend/character.py`](file:///c:/Users/Gilbert/MemoWeave/backend/character.py)

**Function:** `call_reasoning_llm()` (lines 62-93)

**System Prompt** (lines 73-80):
```python
"You are a macro-level story consistency validator.\n"
"Detect which missing characters[actor], tools, or roles during events.\n"
"Some actors could be locations.\n"
"Summarize issues per chapter in human-readable paragraphs.\n"
"Do NOT reference event IDs or sentence IDs.\n"
"Do NOT rewrite the story, only report violations."
```

**User Prompt Builder:** `build_prompt()` (lines 51-60)
- Aggregates events by chapter from the role_completeness.csv
- Formats: `"Chapter {id}: - {event_text} (actor: {actor}, target: {target}, location: {location})"`

**To Fine-tune:**
- Modify the system prompt content in lines 74-79 to change LLM behavior
- Adjust the user prompt format in `build_prompt()` to change data presentation
- Change `MODEL_NAME` (line 17) to use a different model
- Adjust `temperature` (line 84) for more/less creative responses (currently 0.0)

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
