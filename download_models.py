from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    print("huggingface_hub is required. Install with `pip install huggingface_hub`.")
    raise


PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
SENTENCE_TRANSFORMER_ID = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_MODEL_ID = "dbmdz/bert-large-cased-finetuned-conll03-english"
SPACY_MODEL = "en_core_web_sm"
ALLENNLP_SRL_MODEL_URL = "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz"

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_sentence_transformer() -> Tuple[bool, str]:
    target_root = MODELS_DIR / "sentence_transformers"
    ensure_dir(target_root)
    print("=" * 60)
    print(f"Downloading Sentence Transformer model: {SENTENCE_TRANSFORMER_ID}")
    print(f"Target directory: {target_root}")
    print("=" * 60)

    try:
        snapshot_download(
            repo_id=SENTENCE_TRANSFORMER_ID,
            local_dir=target_root,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return True, f"[OK] Sentence Transformer stored in {target_root}"
    except Exception as exc:  # noqa: BLE001
        return False, f"[FAIL] Failed to download Sentence Transformer: {exc}"


def download_huggingface_model() -> Tuple[bool, str]:
    target_root = MODELS_DIR / "huggingface"
    ensure_dir(target_root)
    print("=" * 60)
    print(f"Downloading HuggingFace SRL model: {HUGGINGFACE_MODEL_ID}")
    print(f"Target directory: {target_root}")
    print("=" * 60)

    try:
        snapshot_download(
            repo_id=HUGGINGFACE_MODEL_ID,
            local_dir=target_root,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return True, f"[OK] HuggingFace model stored in {target_root}"
    except Exception as exc:  # noqa: BLE001
        return False, f"[FAIL] Failed to download HuggingFace model: {exc}"


def download_spacy_model() -> Tuple[bool, str]:
    """Install spaCy model (stored in spaCy's standard location)."""
    print("=" * 60)
    print(f"Installing spaCy model: {SPACY_MODEL}")
    print("=" * 60)
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
            check=True,
            capture_output=False,
        )
        return True, "[OK] spaCy model installed (managed by spaCy)"
    except subprocess.CalledProcessError as exc:
        return False, f"[FAIL] Failed to install spaCy model: {exc}"


def download_allennlp_srl_model() -> Tuple[bool, str]:
    """Download AllenNLP SRL model (will be cached by AllenNLP)."""
    print("=" * 60)
    print("Downloading AllenNLP SRL model")
    print(f"Model URL: {ALLENNLP_SRL_MODEL_URL}")
    print("=" * 60)
    print("Note: AllenNLP has dependency conflicts with spaCy 3.x")
    print("      This step is optional and will be skipped if AllenNLP is not available.")
    print("=" * 60)
    
    try:
        from allennlp.predictors.predictor import Predictor
        
        # Set cache directory
        allennlp_model_dir = MODELS_DIR / "allennlp"
        ensure_dir(allennlp_model_dir)
        os.environ["ALLENNLP_CACHE_ROOT"] = str(allennlp_model_dir)
        
        # Load predictor - this will download the model if not cached
        print("Loading predictor (this will download the model if needed)...")
        predictor = Predictor.from_path(ALLENNLP_SRL_MODEL_URL)
        
        # Test the predictor with a simple sentence
        test_result = predictor.predict("The cat sat on the mat.")
        if test_result and 'verbs' in test_result:
            return True, f"[OK] AllenNLP SRL model downloaded and verified (cached in {allennlp_model_dir})"
        else:
            return False, "[FAIL] AllenNLP model loaded but test failed"
            
    except ImportError:
        return False, "[WARN] AllenNLP not installed (optional). To install: pip install allennlp==2.9.3 allennlp-models==2.9.3"
    except Exception as exc:  # noqa: BLE001
        return False, f"[WARN] Failed to download AllenNLP SRL model: {exc} (optional step)"

def main() -> None:
    print("\n" + "=" * 60)
    print("Temporal Memory Layer - Model Download")
    print("=" * 60)
    print("Models will be stored under: models/\n")

    ensure_dir(MODELS_DIR)

    successes = []

    success, message = download_sentence_transformer()
    print(message)
    successes.append(success)

    success, message = download_huggingface_model()
    print(message)
    successes.append(success)

    success, message = download_spacy_model()
    print(message)
    successes.append(success)

    success, message = download_allennlp_srl_model()
    print(message)
    successes.append(success)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    # Count non-optional failures (AllenNLP is optional)
    required_successes = successes[:-1]  # All except last (AllenNLP)
    if all(required_successes):
        if successes[-1]:  # AllenNLP
            print("[OK] All models downloaded successfully.")
        else:
            print("[OK] All required models downloaded successfully.")
            print("[WARN] AllenNLP SRL model download skipped (optional, has dependency conflicts)")
    else:
        print("[WARN] Some required models failed to download. See messages above.")
        if not successes[-1]:
            print("[WARN] AllenNLP SRL model download skipped (optional)")

    print("\nModels directory contents:")
    for sub in MODELS_DIR.iterdir():
        if sub.is_dir():
            print(f" - {sub.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted.")
        sys.exit(1)

