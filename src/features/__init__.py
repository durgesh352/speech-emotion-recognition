"""Feature extraction tools."""
from src.features.audio_features import (
    extract_features_from_audio,
    extract_features_from_file,
    extract_timeline_windows
)
from src.features.intensity_calculator import calculate_emotion_intensity

__all__ = [
    "extract_features_from_audio",
    "extract_features_from_file",
    "extract_timeline_windows",
    "calculate_emotion_intensity"
]
