"""NLP emotion analysis tools."""
from src.nlp.nlp_emotion import analyze_text_emotion, detect_language
from src.nlp.multimodal_fusion import fuse_multimodal_predictions

__all__ = [
    "analyze_text_emotion",
    "detect_language",
    "fuse_multimodal_predictions"
]
