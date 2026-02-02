# pipeline.py

import os
import sys
import traceback
from datetime import datetime
from typing import Optional
from pathlib import Path
import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Handle both module and direct execution
if __name__ == "__main__" and __package__ is None:
    # Running directly - add parent directory to path and use absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backend.step1_text_processing import process_text
    from backend.step2_event_extraction import extract_events
    from backend.step3_temporal_normalization import normalize_temporal_expressions
    from backend.step4_semantic_representation import create_semantic_representations
    from backend.step5_memory_storage import create_memory_module
    from backend.utils import ensure_directory, get_reference_date
else:
    # Running as module - use relative imports
    from .step1_text_processing import process_text
    from .step2_event_extraction import extract_events
    from .step3_temporal_normalization import normalize_temporal_expressions
    from .step4_semantic_representation import create_semantic_representations
    from .step5_memory_storage import create_memory_module
    from .utils import ensure_directory, get_reference_date


def validate_input(input_file: str) -> bool:
    """
    Check if input file exists and is readable.
    
    Args:
        input_file: Path to input text file
        
    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    if not os.path.isfile(input_file):
        print(f"Error: Path is not a file: {input_file}")
        return False
    
    if not os.access(input_file, os.R_OK):
        print(f"Error: Cannot read file: {input_file}")
        return False
    
    return True


def setup_output_directories(output_dir: str) -> None:
    """
    Create necessary output directories.
    
    Args:
        output_dir: Base output directory
    """
    output_dir = Path(output_dir)
    (output_dir / "preprocessed").mkdir(parents=True, exist_ok=True)
    (output_dir / "memory").mkdir(parents=True, exist_ok=True)


def clear_output_directory(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)

    if output_dir.exists() and output_dir.is_dir():
        #print(f"Deleting output directory: {output_dir}")
        shutil.rmtree(output_dir)

def run_pipeline(
    input_file: str,
    output_dir: str = "../output",
    reference_date: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> str:
    output_path = PROJECT_ROOT / output_dir

    clear_output_directory(output_path)

    """
    Orchestrate all 5 steps of the Temporal Memory Layer pipeline.
    
    Args:
        input_file: Path to input text file
        output_dir: Output directory for all processed files
        reference_date: Reference date for temporal normalization (defaults to current date)
        embedding_model: Sentence transformer model name
        
    Returns:
        Path to the final memory_module.json file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If any step fails
    """
    #print("=" * 60)
    #print("Temporal Memory Layer Pipeline")
    #print("=" * 60)
    #print()
    
    # Validate input
    #print("Step 0: Validating input...")
    print("Validating input...")
    if not validate_input(input_file):
        raise FileNotFoundError(f"Invalid input file: {input_file}")
    #print(f"OK: Input file validated: {input_file}")
    #print()
    
    # Setup output directories
    #print("Step 0: Setting up output directories...")
    print("Setting up output directories...")
    setup_output_directories(output_path)
    #print(f"OK: Output directories created: {output_dir}")
    #print()
    
    # Get reference date
    if reference_date is None:
        reference_date = get_reference_date()
    
    step_errors = []
    
    try:
        # Step 1: Text Processing
        #print("=" * 60)
        print()
        print("STEP 1: Text Processing")
        #print("=" * 60)
        #print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Step 1...")
        try:
            text_data = process_text(input_file, output_dir)
            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] OK: Step 1 completed successfully")
            print(f"  - Processed {len(text_data.get('chapters', []))} chapters")
            print(f"  - Processed {len(text_data.get('sentences', []))} sentences")
        except Exception as e:
            error_msg = f"Step 1 (Text Processing) failed: {str(e)}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
            print("\nError details:")
            traceback.print_exc()
            step_errors.append(error_msg)
            raise
        print()
        
        # Step 2: Event & Role Extraction
        #print("=" * 60)
        print("STEP 2: Event & Role Extraction")
        #print("=" * 60)
        #print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Step 2...")
        try:
            events = extract_events(output_dir, output_dir)
            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] OK: Step 2 completed successfully")
            print(f"  - Extracted {len(events)} events")
        except Exception as e:
            error_msg = f"Step 2 (Event Extraction) failed: {str(e)}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
            print("\nError details:")
            traceback.print_exc()
            step_errors.append(error_msg)
            raise
        print()
        
        # Step 3: Temporal Normalization
        #print("=" * 60)
        print("STEP 3: Temporal Normalization")
        #print("=" * 60)
        #print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Step 3...")
        #print(f"  - Using reference date: {reference_date}")
        try:
            timestamps = normalize_temporal_expressions(output_dir, output_dir, reference_date)
            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] OK: Step 3 completed successfully")
            total_expr = timestamps.get("total_expressions", 0)
            print(f"  - Normalized {total_expr} time expressions")
        except Exception as e:
            error_msg = f"Step 3 (Temporal Normalization) failed: {str(e)}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
            print("\nError details:")
            traceback.print_exc()
            step_errors.append(error_msg)
            raise
        print()
        
        # Step 4: Semantic Representation
        #print("=" * 60)
        print("STEP 4: Semantic Representation & Memory Structuring")
        #print("=" * 60)
        #print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Step 4...")
        #print(f"  - Using embedding model: {embedding_model}")
        try:
            semantic_data = create_semantic_representations(output_dir, output_dir, embedding_model)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] OK: Step 4 completed successfully")
            total_events = semantic_data.get("semantic_memory", {}).get("total_events", 0)
            emb_dim = semantic_data.get("semantic_memory", {}).get("embedding_dim", 0)
            #print(f"  - Generated embeddings for {total_events} events")
            #print(f"  - Embedding dimension: {emb_dim}")
        except Exception as e:
            error_msg = f"Step 4 (Semantic Representation) failed: {str(e)}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
            print("\nError details:")
            traceback.print_exc()
            step_errors.append(error_msg)
            raise
        print()
        
        # Step 5: Memory Storage
        #print("=" * 60)
        print("STEP 5: Temporal Memory Storage Layer")
        #print("=" * 60)
        #print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Step 5...")
        try:
            memory_module = create_memory_module(output_dir, output_dir)
            print()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] OK: Step 5 completed successfully")
            metadata = memory_module.get("metadata", {})
            print(f"  - Total chapters: {metadata.get('total_chapters', 0)}")
            print(f"  - Total sentences: {metadata.get('total_sentences', 0)}")
            print(f"  - Total events: {metadata.get('total_events', 0)}")
            print(f"  - Total characters: {metadata.get('total_characters', 0)}")
        except Exception as e:
            error_msg = f"Step 5 (Memory Storage) failed: {str(e)}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
            print("\nError details:")
            traceback.print_exc()
            step_errors.append(error_msg)
            raise
        print()

        # Step 6: Temporal-Causal Reasoning
        #print("=" * 60)
        print("STEP 6: Temporal-Causal Reasoning")
        #print("=" * 60)
        #print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Step 6...")
        try:
            # Import here to avoid circular dependencies if any
            # Import here to avoid circular dependencies if any
            try:
                from .step6_reasoning import process_reasoning
            except ImportError:
                # If running as script/improper module context, fallback
                from backend.step6_reasoning import process_reasoning
            
            reasoning_result = process_reasoning(output_dir, output_dir)
            
            if reasoning_result.get("status") == "success":
                print()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] OK: Step 6 completed successfully")
                data = reasoning_result.get("data", {})
                temp_rels = len(data.get("temporal_relations", []))
                causal_rels = len(data.get("causal_relations", []))
                print(f"  - Inferred {temp_rels} temporal relations")
                print(f"  - Inferred {causal_rels} causal relations")
            else:
                print()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: Step 6 failed or skipped")
                print(f"  - Error: {reasoning_result.get('error')}")
                
        except Exception as e:
            error_msg = f"Step 6 (Reasoning) failed: {str(e)}"
            #print(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: {error_msg}")
            print("  - Continuing (Step 6 is optional/experimental)")
            traceback.print_exc()
        print()
        
        # Final output path
        final_output_path = os.path.join(output_dir, "memory", "memory_module.json")
        
        #print("=" * 60)
        print("Temporal Memory Pipeline Completed Successfully!")
        #print("=" * 60)
        # print(f"[{datetime.now().strftime('%H:%M:%S')}] Final memory module: {final_output_path}")
        # print()
        # print("Output files created:")
        # print(f"  - {output_dir}/preprocessed/chapters/ (individual chapter files)")
        # print(f"  - {output_dir}/preprocessed/chapters_index.json")
        # print(f"  - {output_dir}/preprocessed/sentences.json")
        # print(f"  - {output_dir}/memory/events.json")
        # print(f"  - {output_dir}/memory/timestamps.json")
        # print(f"  - {output_dir}/memory/event_embeddings.json")
        # print(f"  - {output_dir}/memory/memory_semantic.json")
        # print(f"  - {output_dir}/memory/memory_module.json")
        # print(f"  - {output_dir}/memory/reasoning_graph.json")
        # print()
        
        return final_output_path
        
    except Exception as e:
        print()
        print("=" * 60)
        print("Temporal Memory Pipeline Failed!")
        print("=" * 60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error occurred during pipeline execution")
        print()
        if step_errors:
            print("Failed steps:")
            for i, error in enumerate(step_errors, 1):
                print(f"  {i}. {error}")
            print()
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        print()
        raise RuntimeError(f"Pipeline execution failed at step: {step_errors[-1] if step_errors else 'Unknown'}") from e


def main():
    """
    Main entry point for command-line usage.
    """
    if len(sys.argv) < 2:
        print("=" * 60)
        print("Temporal Memory Layer Pipeline")
        print("=" * 60)
        print()
        print("ERROR: No input file provided!")
        print()
        print("Usage:")
        print("  python -m backend.pipeline <input_file> [output_dir] [reference_date] [embedding_model]")
        print("  OR")
        print("  python backend/pipeline.py <input_file> [output_dir] [reference_date] [embedding_model]")
        print()
        print("Arguments:")
        print("  input_file      Path to input text file (REQUIRED)")
        print("  output_dir      Output directory (optional, default: 'output')")
        print("  reference_date  Reference date for normalization in YYYY-MM-DD format (optional, default: current date)")
        print("  embedding_model Sentence transformer model name (optional, default: 'all-MiniLM-L6-v2')")
        print()
        print("Example:")
        print("  python -m backend.pipeline data/story.txt")
        print("  python -m backend.pipeline data/story.txt output 2025-01-15")
        print()
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    reference_date = sys.argv[3] if len(sys.argv) > 3 else None
    embedding_model = sys.argv[4] if len(sys.argv) > 4 else "all-MiniLM-L6-v2"
    
    try:
        result_path = run_pipeline(input_file, output_dir, reference_date, embedding_model)
        #print(f"\n{'='*60}")
        print("Operation successful!")
        #print(f"{'='*60}")
        #print(f"Memory module saved to: {result_path}")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print("FILE NOT FOUND ERROR")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nPlease check:")
        print(f"  1. Does the file exist? {input_file}")
        print(f"  2. Is the path correct?")
        print(f"  3. Do you have read permissions?")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n{'='*60}")
        print("RUNTIME ERROR")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nCommon issues:")
        print("  1. Missing models - Run: python download_models.py")
        print("  2. Missing dependencies - Run: pip install -r requirements.txt")
        print("  3. Check the error details above to see which step failed")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*60}")
        print("UNEXPECTED ERROR")
        print(f"{'='*60}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()