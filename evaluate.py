"""
Model Evaluation and Metrics Viewer
Displays model classification report, confusion matrix, and tests a single audio file.
"""

import sys
import json
import argparse
from pathlib import Path

# Fix Windows console encoding for Unicode / Emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from src.config import METADATA_FILE, EMOTIONS
from src.models.ser_pipeline import SpeechEmotionRecognizer

def show_model_metrics():
    if not METADATA_FILE.exists():
        print("No metadata found. Please train the model first with `python train.py`.")
        return

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("=" * 60)
    print(" 📊 SPEECH EMOTION RECOGNITION - MODEL METRICS")
    print("=" * 60)
    print(f"Overall Accuracy:  {meta.get('accuracy', 0) * 100:.2f}%")
    print(f"Total Samples:     {meta.get('total_samples', 0)}")
    print(f"Train/Test Split:  {meta.get('train_samples', 0)} / {meta.get('test_samples', 0)}")
    print(f"Feature Vector:    {meta.get('num_features', 0)} dimensions")
    print(f"Languages:         {', '.join(meta.get('languages_supported', []))}")

    print("\nPer-Emotion Detailed Performance:")
    print(f"{'Emotion':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 58)
    report = meta.get("classification_report", {})
    for emo in EMOTIONS:
        d = report.get(emo, {})
        p = f"{d.get('precision', 0)*100:.1f}%"
        r = f"{d.get('recall', 0)*100:.1f}%"
        f1 = f"{d.get('f1-score', 0)*100:.1f}%"
        s = f"{int(d.get('support', 0))}"
        print(f"{emo:<12} | {p:<10} | {r:<10} | {f1:<10} | {s:<8}")

    print("\nConfusion Matrix:")
    cm = meta.get("confusion_matrix", [])
    print(f"{'':<10}" + "".join([f"{e[:4]:>7}" for e in EMOTIONS]))
    for idx, row in enumerate(cm):
        row_str = "".join([f"{val:>7}" for val in row])
        print(f"{EMOTIONS[idx]:<10}{row_str}")

def test_single_audio(audio_path: str, transcript: str = None):
    p = Path(audio_path)
    if not p.exists():
        print(f"File not found: {audio_path}")
        return

    recognizer = SpeechEmotionRecognizer()
    print(f"\nAnalyzing: {p.name}...")
    res = recognizer.predict_audio(p, transcript=transcript)

    print("=" * 50)
    print(f" Predicted Emotion:    {res['predicted_emotion']}")
    print(f" Confidence Score:     {res['confidence_percentage']}%")
    print(f" Emotion Intensity:    {res['emotion_intensity']['level_en']} ({res['emotion_intensity']['intensity_percentage']}%)")
    print(f" Audio Duration:       {res['duration_seconds']}s")
    print("=" * 50)
    print("\nProbability Breakdown:")
    for emo, prob in res['emotion_probabilities'].items():
        bar = "█" * int(prob * 30)
        print(f"  {emo:<10} [{prob*100:5.1f}%] {bar}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SER model")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file to test")
    parser.add_argument("--transcript", type=str, default=None, help="Optional speech transcript")
    args = parser.parse_args()

    if args.audio:
        test_single_audio(args.audio, args.transcript)
    else:
        show_model_metrics()
