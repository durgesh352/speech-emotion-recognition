"""
Speech Emotion Recognition Training CLI
Trains the bilingual (Hindi + English) emotion classifier.

Usage:
    python train.py
    python train.py --samples 80
    python train.py --dataset_dir path/to/wav/files
"""

import sys
import argparse
from pathlib import Path

# Fix Windows console encoding for Unicode / Emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from src.models.model_trainer import SpeechEmotionTrainer
from src.utils.audio_generator import generate_sample_audio_files
from src.config import SAMPLE_AUDIO_DIR

def main():
    parser = argparse.ArgumentParser(description="Train Bilingual Speech Emotion Recognition Model")
    parser.add_argument(
        "--samples",
        type=int,
        default=60,
        help="Number of synthetic samples per emotion (bilingual EN + HI)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Optional path to directory with real WAV files (e.g., RAVDESS, CREMA-D, or custom folder)"
    )
    parser.add_argument(
        "--no_sample_gen",
        action="store_true",
        help="Skip generating preset sample audio files"
    )

    args = parser.parse_args()

    print("=" * 65)
    print(" 🎙️  SPEECH EMOTION RECOGNITION (SER) - BILINGUAL MODEL TRAINER")
    print(" Languages: English (EN) & Hindi (HI) | 7 Core Emotions")
    print("=" * 65)

    trainer = SpeechEmotionTrainer()

    if args.dataset_dir:
        print(f"\n[1/3] Loading real-world audio dataset from: {args.dataset_dir}")
        X, y = trainer.load_dataset_from_directory(args.dataset_dir)
        if len(X) == 0:
            print("No valid audio files found in directory! Falling back to synthetic bilingual generator.")
            X, y = trainer.generate_bilingual_training_dataset(samples_per_emotion=args.samples)
    else:
        print(f"\n[1/3] Generating bilingual acoustic training dataset ({args.samples} samples/emotion)...")
        X, y = trainer.generate_bilingual_training_dataset(samples_per_emotion=args.samples)

    print(f"\n[2/3] Training Ensemble Classifier (MLP Neural Net + Random Forest) on {len(X)} samples...")
    metadata = trainer.train(X, y)

    print("\n" + "=" * 65)
    print(" 🎯 TRAINING REPORT SUMMARY")
    print("=" * 65)
    print(f" Test Accuracy:     {metadata['accuracy'] * 100:.2f}%")
    print(f" Total Samples:     {metadata['total_samples']}")
    print(f" Train Samples:     {metadata['train_samples']}")
    print(f" Test Samples:      {metadata['test_samples']}")
    print(f" Features Extracted: {metadata['num_features']}")
    print("\nPer-Emotion Breakdown:")
    for emo in trainer.emotions:
        score_data = metadata['classification_report'].get(emo, {})
        f1 = score_data.get('f1-score', 0.0) * 100
        prec = score_data.get('precision', 0.0) * 100
        rec = score_data.get('recall', 0.0) * 100
        print(f"  - {emo:<10}: F1 = {f1:.1f}% | Precision = {prec:.1f}% | Recall = {rec:.1f}%")

    if not args.no_sample_gen:
        print(f"\n[3/3] Generating preset bilingual sample audios in {SAMPLE_AUDIO_DIR}...")
        files = generate_sample_audio_files()
        print(f" Generated {len(files['english'])} English and {len(files['hindi'])} Hindi sample audio files.")

    print("\n✅ Training complete! Models and assets are ready for inference and dashboard deployment.")

if __name__ == "__main__":
    main()
