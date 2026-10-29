"""
End-to-End Speech Emotion Recognition Inference Pipeline
Provides full inference:
- Primary Emotion Classification
- Confidence Score calculation
- Emotion Intensity quantification (Acoustic Arousal + Valence)
- Dynamic Emotion Timeline (Time-series emotion progression)
- NLP Multimodal Fusion (English & Hindi)
"""

import io
import json
import numpy as np
import librosa
import soundfile as sf
import joblib
from pathlib import Path

from src.config import (
    MODEL_FILE,
    SCALER_FILE,
    METADATA_FILE,
    EMOTIONS,
    SAMPLE_RATE,
    WINDOW_SIZE_SEC,
    HOP_SIZE_SEC
)
from src.features.audio_features import (
    extract_features_from_audio,
    extract_timeline_windows
)
from src.features.intensity_calculator import calculate_emotion_intensity
from src.nlp.nlp_emotion import analyze_text_emotion
from src.nlp.multimodal_fusion import fuse_multimodal_predictions


class SpeechEmotionRecognizer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.emotions = EMOTIONS
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        """Loads trained model and scaler if available."""
        if MODEL_FILE.exists() and SCALER_FILE.exists():
            try:
                self.model = joblib.load(MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
                if METADATA_FILE.exists():
                    with open(METADATA_FILE, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                self.is_loaded = True
            except Exception as e:
                print(f"Error loading trained model: {e}")
                self.is_loaded = False
        else:
            self.is_loaded = False

    def predict_audio(
        self,
        audio_input: str | Path | bytes | io.BytesIO | np.ndarray,
        sr: int = SAMPLE_RATE,
        language: str = "auto",
        transcript: str | None = None,
        audio_weight: float = 0.70
    ) -> dict:
        """
        Runs complete emotion prediction, intensity calculation, timeline segmentation,
        and multimodal NLP fusion.
        """
        if not self.is_loaded:
            self._load_model()
            if not self.is_loaded:
                raise RuntimeError(
                    "Model not found. Please train the model first by running `python train.py`."
                )

        # 1. Load and decode audio input into waveform y
        if isinstance(audio_input, (str, Path)):
            y, sample_rate = librosa.load(audio_input, sr=sr, mono=True)
        elif isinstance(audio_input, (bytes, io.BytesIO)):
            if isinstance(audio_input, bytes):
                audio_input = io.BytesIO(audio_input)
            y, sample_rate = sf.read(audio_input)
            if y.ndim > 1:
                y = np.mean(y, axis=1)  # Convert stereo to mono
            if sample_rate != sr:
                y = librosa.resample(y, orig_sr=sample_rate, target_sr=sr)
        elif isinstance(audio_input, np.ndarray):
            y = audio_input.astype(np.float32)
            sample_rate = sr
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        duration = float(len(y)) / sr

        # 2. Extract global features & calculate Emotion Intensity
        global_features = extract_features_from_audio(y, sr=sr)
        intensity_info = calculate_emotion_intensity(y, sr=sr)

        # 3. Model Prediction (Acoustic Probabilities)
        scaled_feat = self.scaler.transform(global_features.reshape(1, -1))
        probabilities_raw = self.model.predict_proba(scaled_feat)[0]
        
        # Map probabilities to classes
        classes = list(self.model.classes_)
        acoustic_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probabilities_raw)}
        
        # Ensure all standard emotions exist in dictionary
        for emo in self.emotions:
            if emo not in acoustic_probs:
                acoustic_probs[emo] = 0.0

        # Sort dominant emotion
        dominant_acoustic_emo = max(acoustic_probs, key=acoustic_probs.get)
        dominant_confidence = acoustic_probs[dominant_acoustic_emo]

        # 4. Multimodal NLP Text Analysis (if transcript given or detected)
        nlp_result = None
        multimodal_result = None
        if transcript and transcript.strip():
            nlp_result = analyze_text_emotion(transcript, forced_lang=language)
            multimodal_result = fuse_multimodal_predictions(
                acoustic_probs=acoustic_probs,
                nlp_probs=nlp_result["emotion_distribution"],
                audio_weight=audio_weight
            )
            final_emotion = multimodal_result["fused_emotion"]
            final_confidence = multimodal_result["confidence"]
            final_distribution = multimodal_result["distribution"]
        else:
            final_emotion = dominant_acoustic_emo
            final_confidence = dominant_confidence
            final_distribution = acoustic_probs

        # 5. Emotion Timeline Segmentation (Sliding Window analysis)
        timeline_windows = extract_timeline_windows(
            y, sr=sr, window_sec=WINDOW_SIZE_SEC, hop_sec=HOP_SIZE_SEC
        )
        timeline_data = []

        for win in timeline_windows:
            win_scaled = self.scaler.transform(win["features"].reshape(1, -1))
            win_probs_raw = self.model.predict_proba(win_scaled)[0]
            win_probs = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, win_probs_raw)}
            
            # Segment emotion intensity
            win_intensity = calculate_emotion_intensity(win["audio"], sr=sr)
            win_dominant = max(win_probs, key=win_probs.get)

            row = {
                "window_idx": win["window_idx"],
                "start_time": win["start_time"],
                "end_time": win["end_time"],
                "center_time": win["center_time"],
                "dominant_emotion": win_dominant,
                "confidence": win_probs[win_dominant],
                "intensity_score": win_intensity["intensity_score"],
                "intensity_percentage": win_intensity["intensity_percentage"],
                "intensity_level": win_intensity["level_en"]
            }
            # Append each emotion probability for timeline charting
            for emo in self.emotions:
                row[f"prob_{emo.lower()}"] = win_probs.get(emo, 0.0)

            timeline_data.append(row)

        return {
            "predicted_emotion": final_emotion,
            "confidence_score": round(final_confidence, 4),
            "confidence_percentage": round(final_confidence * 100, 1),
            "emotion_probabilities": final_distribution,
            "acoustic_probabilities": acoustic_probs,
            "emotion_intensity": intensity_info,
            "timeline": timeline_data,
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "audio_waveform": y,
            "nlp_analysis": nlp_result,
            "multimodal_result": multimodal_result
        }
