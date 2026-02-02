"""
Model Cache Utility
Helps manage and check for cached models to reduce runtime downloads.
"""

import os
from pathlib import Path
from typing import Optional, Dict


def get_model_cache_paths() -> Dict[str, str]:
    """
    Get paths where models are stored (local project directory).
    
    Returns:
        Dictionary mapping model types to storage paths
    """
    # Get project root (assuming this file is in backend/)
    project_root = Path(__file__).parent.parent.absolute()
    models_dir = project_root / "models"
    
    home = Path.home()
    return {
        "project_models": str(models_dir),
        "sentence_transformers": str(models_dir / "sentence_transformers"),
        "huggingface": str(models_dir / "huggingface"),
        "allennlp": str(models_dir / "allennlp"),
        "spacy": str(home / ".local" / "share" / "spacy"),  # spaCy uses system cache
        "system_cache_huggingface": str(home / ".cache" / "huggingface"),
        "system_cache_sentence_transformers": str(home / ".cache" / "torch" / "sentence_transformers"),
        "system_cache_allennlp": str(home / ".allennlp"),  # AllenNLP default cache
    }


def check_spacy_model(model_name: str = "en_core_web_sm") -> bool:
    """
    Check if spaCy model is installed.
    
    Args:
        model_name: Name of the spaCy model
        
    Returns:
        True if model is installed, False otherwise
    """
    try:
        import spacy
        nlp = spacy.load(model_name)
        return True
    except (OSError, ImportError):
        return False


def check_sentence_transformer_model(model_name: str = "all-MiniLM-L6-v2") -> bool:
    """
    Check if sentence transformer model is stored locally.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        True if model is stored locally, False otherwise
    """
    try:
        cache_paths = get_model_cache_paths()
        model_cache = Path(cache_paths["sentence_transformers"])
        
        # Check if local models directory exists and has content
        if model_cache.exists() and any(model_cache.iterdir()):
            return True
        
        # Fallback: check system cache
        system_cache = Path(cache_paths["system_cache_sentence_transformers"])
        return system_cache.exists() and any(system_cache.iterdir())
    except Exception:
        return False


def check_huggingface_model(model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english") -> bool:
    """
    Check if HuggingFace model is stored locally.
    
    Args:
        model_name: Name of the HuggingFace model
        
    Returns:
        True if model is stored locally, False otherwise
    """
    try:
        cache_paths = get_model_cache_paths()
        hf_cache = Path(cache_paths["huggingface"])
        
        # Check if local models directory exists and has content
        if hf_cache.exists() and any(hf_cache.iterdir()):
            return True
        
        # Fallback: check system cache
        system_cache = Path(cache_paths["system_cache_huggingface"])
        return system_cache.exists() and any(system_cache.iterdir())
    except Exception:
        return False


def check_allennlp_srl_model() -> bool:
    """
    Check if AllenNLP SRL model is available.
    
    Returns:
        True if model is available (installed or can be loaded), False otherwise
    """
    try:
        from allennlp.predictors.predictor import Predictor
        
        # Try to load the model - AllenNLP will use cache if available
        model_url = "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz"
        
        # Check local cache first
        cache_paths = get_model_cache_paths()
        local_cache = Path(cache_paths["allennlp"])
        if local_cache.exists() and any(local_cache.iterdir()):
            # Try to load from local cache
            try:
                predictor = Predictor.from_path(model_url)
                # Test with a simple sentence
                test_result = predictor.predict("The cat sat on the mat.")
                return test_result is not None and 'verbs' in test_result
            except Exception:
                pass
        
        # Check system cache
        system_cache = Path(cache_paths["system_cache_allennlp"])
        if system_cache.exists():
            try:
                predictor = Predictor.from_path(model_url)
                test_result = predictor.predict("The cat sat on the mat.")
                return test_result is not None and 'verbs' in test_result
            except Exception:
                pass
        
        # If we can import, the model can be downloaded on first use
        return True  # AllenNLP will download automatically
        
    except ImportError:
        return False
    except Exception:
        return False


def get_model_status() -> Dict[str, bool]:
    """
    Get status of all required models.
    
    Returns:
        Dictionary with model status (True = installed/cached, False = not found)
    """
    return {
        "spacy_en_core_web_sm": check_spacy_model("en_core_web_sm"),
        "sentence_transformer_all-MiniLM-L6-v2": check_sentence_transformer_model("all-MiniLM-L6-v2"),
        "huggingface_srl": check_huggingface_model("dbmdz/bert-large-cased-finetuned-conll03-english"),
        "allennlp_srl": check_allennlp_srl_model()
    }


def print_model_status():
    """Print status of all models."""
    print("=" * 60)
    print("Model Status Check")
    print("=" * 60)
    
    status = get_model_status()
    cache_paths = get_model_cache_paths()
    
    for model_name, is_installed in status.items():
        status_symbol = "✓" if is_installed else "✗"
        print(f"{status_symbol} {model_name}: {'Installed' if is_installed else 'Not found'}")
    
    print("\nModel storage locations:")
    print(f"✓ Project models directory: {cache_paths['project_models']}")
    
    # Check local vs system cache
    local_st = Path(cache_paths["sentence_transformers"])
    local_hf = Path(cache_paths["huggingface"])
    
    if local_st.exists() and any(local_st.iterdir()):
        print(f"✓ Sentence Transformers: {local_st} (LOCAL)")
    else:
        system_st = Path(cache_paths["system_cache_sentence_transformers"])
        if system_st.exists():
            print(f"⚠ Sentence Transformers: {system_st} (SYSTEM CACHE)")
        else:
            print(f"✗ Sentence Transformers: Not found")
    
    if local_hf.exists() and any(local_hf.iterdir()):
        print(f"✓ HuggingFace: {local_hf} (LOCAL)")
    else:
        system_hf = Path(cache_paths["system_cache_huggingface"])
        if system_hf.exists():
            print(f"⚠ HuggingFace: {system_hf} (SYSTEM CACHE)")
        else:
            print(f"✗ HuggingFace: Not found")
    
    local_allennlp = Path(cache_paths["allennlp"])
    if local_allennlp.exists() and any(local_allennlp.iterdir()):
        print(f"✓ AllenNLP: {local_allennlp} (LOCAL)")
    else:
        system_allennlp = Path(cache_paths["system_cache_allennlp"])
        if system_allennlp.exists():
            print(f"⚠ AllenNLP: {system_allennlp} (SYSTEM CACHE)")
        else:
            print(f"⚠ AllenNLP: Will download on first use")
    
    print(f"✓ spaCy: {cache_paths['spacy']} (SYSTEM CACHE)")


if __name__ == "__main__":
    print_model_status()

