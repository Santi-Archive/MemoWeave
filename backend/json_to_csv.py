# json_to_csv.py
import json
import csv
import os
from typing import Callable


# =========================
# LOGGING
# =========================

def log(msg: str, callback: Callable = None):
    formatted = msg
    if callback:
        callback(formatted)
    else:
        print(formatted)


# =========================
# MAIN ENTRY POINT
# =========================

def run_json_to_csv(
    memory_module_path: str,
    rule_class: str,
    log_callback: Callable = None
):
    """
    Projects memory_module.json into a rule-specific CSV.

    IMPORTANT:
    - CSVs are LLM-facing
    - Rule logic stays in Python
    """

    if not os.path.exists(memory_module_path):
        raise FileNotFoundError(f"Memory module not found: {memory_module_path}")

    with open(memory_module_path, "r", encoding="utf-8") as f:
        memory = json.load(f)

    events = memory.get("events", [])
    if not events:
        log("No events found in memory module.", log_callback)
        return None

    output_dir = os.path.join("output", "memory")
    os.makedirs(output_dir, exist_ok=True)

    # ======================================================
    # TEMPORAL CONSISTENCY — LLM-FACING
    # ======================================================
    if rule_class == "temporal":
        csv_path = os.path.join(output_dir, "temporal_consistency.csv")

        fieldnames = [
            "event_id",
            "chapter_id",
            "sentence_id",
            "action_lemma",
            "time_raw",
            "time_normalized",
            "time_type",
            "text"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ev in events:
                time_type = ev.get("time_type")

                # ⛔ Skip rows with no temporal signal
                if time_type is None:
                    continue

                writer.writerow({
                    "event_id": ev.get("event_id"),
                    "chapter_id": ev.get("chapter_id"),
                    "sentence_id": ev.get("sentence_id"),
                    "action_lemma": ev.get("action_lemma"),
                    "time_raw": ev.get("time_raw"),
                    "time_normalized": ev.get("time_normalized"),
                    "time_type": time_type,
                    "text": ev.get("text"),
                })

        log("You selected Temporal Consistencies", log_callback)
        return csv_path

    # ======================================================
    # ROLE COMPLETENESS — LLM-FACING
    # ======================================================
    elif rule_class == "role_completeness":
        csv_path = os.path.join(output_dir, "role_completeness.csv")

        fieldnames = [
            "event_id",
            "chapter_id",
            "sentence_id",
            "action_lemma",
            "actor",
            "target",
            "location",
            "text",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ev in events:
                actor = ev.get("actor")

                # ⛔ Skip rows with no actor
                if actor is None:
                    continue

                writer.writerow({
                    "event_id": ev.get("event_id"),
                    "chapter_id": ev.get("chapter_id"),
                    "sentence_id": ev.get("sentence_id"),
                    "action_lemma": ev.get("action_lemma"),
                    "actor": actor,
                    "target": ev.get("target") or "",
                    "location": ev.get("location") or "",
                    "text": ev.get("text"),
                })

        log("You selected Role Completeness", log_callback)
        return csv_path

    # ======================================================
    # UNKNOWN RULE
    # ======================================================
    else:
        raise ValueError(f"Unknown rule class: {rule_class}")