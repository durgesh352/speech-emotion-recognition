"""
Bilingual Speech Emotion Recognition Model Trainer
Supports training on synthesized bilingual datasets (Hindi + English) as well as
loading real-world speech emotion datasets (e.g., RAVDESS, CREMA-D, SAVEE, or custom folders).
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

from src.config import (
    EMOTIONS,
    MODEL_FILE,
    SCALER_FILE,
    METADATA_FILE,
    SAMPLE_RATE
)
from src.features.audio_features import extract_features_from_audio, extract_features_from_file
from src.utils.audio_generator import synthesize_emotional_speech

# Map standard RAVDESS emotion codes to our standard 7 emotions
RAVDESS_EMOTION_MAP = {
    "01": "Neutral",
    "02": "Neutral",  # calm mapped to neutral
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}


class SpeechEmotionTrainer:
    def __init__(self):
        self.emotions = EMOTIONS
        self.scaler = StandardScaler()
        # High-capacity ensemble classifier: MLP Neural Network + Random Forest
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
        self.rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            random_state=42,
            n_jobs=-1
        )
        self.model = VotingClassifier(
            estimators=[("mlp", self.mlp), ("rf", self.rf)],
            voting="soft"
        )
        self.metadata = {}

    def generate_bilingual_training_dataset(
        self,
        samples_per_emotion: int = 50,
        augment: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes a rich bilingual training dataset covering both English
        and Hindi acoustic nuances across all 7 emotions with prosodic variations.
        """
        X = []
        y = []

        print(f"Generating bilingual dataset: {samples_per_emotion} samples per emotion (English + Hindi)...")

        for emo_idx, emo in enumerate(self.emotions):
            # Equal proportion of English and Hindi acoustic samples
            en_count = samples_per_emotion // 2
            hi_count = samples_per_emotion - en_count

            # English audio synthesis with acoustic variation
            for _ in range(en_count):
                dur = np.random.uniform(2.5, 3.5)
                audio, sr = synthesize_emotional_speech(emo, language="en", duration=dur)
                
                # Feature extraction
                feat = extract_features_from_audio(audio, sr=sr)
                X.append(feat)
                y.append(emo)

                # Augmentation: pitch shift / noise injection
                if augment and np.random.rand() > 0.4:
                    noisy = audio + np.random.normal(0, 0.02, len(audio))
                    feat_aug = extract_features_from_audio(noisy, sr=sr)
                    X.append(feat_aug)
                    y.append(emo)

            # Hindi audio synthesis with Hindi prosodic characteristics
            for _ in range(hi_count):
                dur = np.random.uniform(2.5, 3.5)
                audio, sr = synthesize_emotional_speech(emo, language="hi", duration=dur)
                
                feat = extract_features_from_audio(audio, sr=sr)
                X.append(feat)
                y.append(emo)

                # Augmentation
                if augment and np.random.rand() > 0.4:
                    scaled_audio = audio * np.random.uniform(0.8, 1.2)
                    feat_aug = extract_features_from_audio(scaled_audio, sr=sr)
                    X.append(feat_aug)
                    y.append(emo)

        return np.array(X, dtype=np.float32), np.array(y)

    def load_dataset_from_directory(self, data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads and extracts features from custom WAV files stored in a directory.
        Supports folder-based labeling (e.g. data_dir/Angry/*.wav) or RAVDESS filenames.
        """
        data_path = Path(data_dir)
        X = []
        y = []

        if not data_path.exists():
            raise FileNotFoundError(f"Directory {data_dir} does not exist.")

        # Search for .wav and .mp3 files
        audio_files = list(data_path.rglob("*.wav")) + list(data_path.rglob("*.mp3"))

        for file_path in audio_files:
            emotion_label = None
            parent_dir_name = file_path.parent.name.capitalize()

            # Check if parent directory matches one of our emotions
            for emo in self.emotions:
                if emo.lower() in parent_dir_name.lower():
                    emotion_label = emo
                    break

            # Check if filename is RAVDESS format (e.g., 03-01-05-01-02-01-01.wav)
            if not emotion_label:
                parts = file_path.stem.split("-")
                if len(parts) >= 3 and parts[2] in RAVDESS_EMOTION_MAP:
                    emotion_label = RAVDESS_EMOTION_MAP[parts[2]]

            # Check if filename contains emotion word directly
            if not emotion_label:
                for emo in self.emotions:
                    if emo.lower() in file_path.stem.lower():
                        emotion_label = emo
                        break

            if emotion_label and emotion_label in self.emotions:
                try:
                    feat, _, _ = extract_features_from_file(file_path)
                    X.append(feat)
                    y.append(emotion_label)
                except Exception as e:
                    print(f"Skipping {file_path.name}: {e}")

        return np.array(X, dtype=np.float32), np.array(y)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.20,
        save_artifacts: bool = True
    ) -> dict:
        """
        Scales features, trains the bilingual ensemble model, computes validation metrics,
        and saves artifacts to disk.
        """
        if len(X) == 0:
            raise ValueError("Dataset is empty. Cannot train model.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"Training dataset: {len(X_train)} samples, Test dataset: {len(X_test)} samples")
        print(f"Feature vector dimensionality: {X_train.shape[1]}")

        # Standard Scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Fit model
        print("Fitting bilingual Speech Emotion Ensemble (MLP + Random Forest)...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, target_names=self.emotions, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=self.emotions).tolist()

        print(f"Test Accuracy: {acc * 100:.2f}%")

        self.metadata = {
            "accuracy": round(acc, 4),
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "num_features": int(X_train.shape[1]),
            "emotions": self.emotions,
            "classification_report": report,
            "confusion_matrix": cm,
            "languages_supported": ["en", "hi"]
        }

        if save_artifacts:
            MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, MODEL_FILE)
            joblib.dump(self.scaler, SCALER_FILE)
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            print(f"Model saved to {MODEL_FILE}")
            print(f"Scaler saved to {SCALER_FILE}")

        return self.metadata
