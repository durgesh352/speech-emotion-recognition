"""
Speech Emotion Recognition Configuration
Supports Bilingual Processing (Hindi and English), Emotion Intensity, and Multi-modal NLP.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_AUDIO_DIR = DATA_DIR / "sample_audios"
MODELS_DIR = BASE_DIR / "saved_models"

# Standard 7 Core Emotions
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Emotion Visual Metadata (Color, Emoji, Description)
EMOTION_META = {
    "Angry": {
        "emoji": "😡",
        "color": "#E53E3E",
        "hi_name": "गुस्सा (Angry)",
        "description": "High vocal arousal, sharp pitch peaks, high acoustic energy"
    },
    "Disgust": {
        "emoji": "🤢",
        "color": "#805AD5",
        "hi_name": "घृणा / चिढ़ (Disgust)",
        "description": "Lower pitch variability, elongated vowel formants, guttural tone"
    },
    "Fear": {
        "emoji": "😨",
        "color": "#DD6B20",
        "hi_name": "डर / भय (Fear)",
        "description": "Rapid frequency jitter, elevated mean pitch, breathy vocalics"
    },
    "Happy": {
        "emoji": "😊",
        "color": "#38A169",
        "hi_name": "खुशी / आनंद (Happy)",
        "description": "Dynamic pitch range, elevated brightness, lively rhythm"
    },
    "Neutral": {
        "emoji": "😐",
        "color": "#718096",
        "hi_name": "शांत / सामान्य (Neutral)",
        "description": "Steady pitch contour, moderate energy, flat spectral tilt"
    },
    "Sad": {
        "emoji": "😢",
        "color": "#3182CE",
        "hi_name": "उदासी / गम (Sad)",
        "description": "Depressed energy, falling pitch slope, subdued spectral energy"
    },
    "Surprise": {
        "emoji": "😲",
        "color": "#D69E2E",
        "hi_name": "आश्चर्य / हैरान (Surprise)",
        "description": "Sudden upward pitch inflection, rapid onset of spectral energy"
    }
}

# Supported Languages
LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "auto": "Auto-Detect"
}

# Audio Processing Parameters
SAMPLE_RATE = 22050
DURATION = 3.0  # target duration in seconds for standardized chunk
N_MFCC = 40     # Number of MFCC coefficients
N_MELS = 128    # Mel filter banks
N_FFT = 2048
HOP_LENGTH = 512

# Timeline Segmentation Parameters
WINDOW_SIZE_SEC = 2.0   # Sliding window chunk size in seconds
HOP_SIZE_SEC = 0.5      # Sliding window step size in seconds

# Emotion Intensity Thresholds (0.0 to 1.0)
INTENSITY_LEVELS = [
    (0.0, 0.35, "Subtle", "हल्का"),
    (0.35, 0.65, "Moderate", "मध्यम"),
    (0.65, 0.85, "High", "तीव्र"),
    (0.85, 1.00, "Intense", "अत्यधिक तीव्र")
]

# Model files
MODEL_FILE = MODELS_DIR / "ser_bilingual_model.joblib"
SCALER_FILE = MODELS_DIR / "ser_scaler.joblib"
METADATA_FILE = MODELS_DIR / "ser_metadata.json"
