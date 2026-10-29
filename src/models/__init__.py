"""Speech Emotion Recognition Models and Inference Pipeline."""
from src.models.model_trainer import SpeechEmotionTrainer
from src.models.ser_pipeline import SpeechEmotionRecognizer

__all__ = ["SpeechEmotionTrainer", "SpeechEmotionRecognizer"]
