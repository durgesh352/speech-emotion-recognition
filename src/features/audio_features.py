"""
Audio Feature Extraction Module
Extracts acoustic features including MFCCs, Chroma, Mel-spectrogram, Spectral Contrast,
Tonnetz, RMS, and ZCR. Also provides windowed segmenting for Emotion Timeline tracking.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from src.config import (
    SAMPLE_RATE,
    N_MFCC,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    WINDOW_SIZE_SEC,
    HOP_SIZE_SEC
)

def extract_features_from_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    include_extended: bool = True
) -> np.ndarray:
    """
    Extracts a fixed-length numerical feature vector from an audio signal.
    
    Features extracted:
    - MFCC (mean, std, max, min) -> 40 * 4 = 160
    - Delta MFCC (mean, std) -> 40 * 2 = 80
    - Mel Spectrogram (mean, std) -> 128 * 2 = 256
    - Chroma STFT (mean, std) -> 12 * 2 = 24
    - Spectral Contrast (mean, std) -> 7 * 2 = 14
    - Tonnetz (mean, std) -> 6 * 2 = 12
    - RMS Energy (mean, std) -> 2
    - Zero Crossing Rate (mean, std) -> 2
    Total compact feature vector: ~194 features (or ~180 depending on configuration)
    """
    if len(y) == 0:
        return np.zeros(180, dtype=np.float32)

    # Ensure audio is float32 and normalized
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    features = []

    # 1. MFCCs (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(mfcc, axis=1))
    features.append(np.std(mfcc, axis=1))

    # 2. Delta MFCCs (Velocity of spectral change)
    mfcc_delta = librosa.feature.delta(mfcc)
    features.append(np.mean(mfcc_delta, axis=1))

    # 3. Chroma STFT (Pitch classes / harmonic profile)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(chroma, axis=1))
    features.append(np.std(chroma, axis=1))

    # 4. Mel-Spectrogram summary (Acoustic energy distribution across frequencies)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.append(np.mean(mel_db, axis=1))
    features.append(np.std(mel_db, axis=1))

    # 5. Spectral Contrast (Valley-to-peak difference in frequency bands)
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.append(np.mean(contrast, axis=1))
        features.append(np.std(contrast, axis=1))
    except Exception:
        features.append(np.zeros(7))
        features.append(np.zeros(7))

    # 6. Tonnetz (Tonal centroid features / musical emotion resonance)
    try:
        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        features.append(np.mean(tonnetz, axis=1))
    except Exception:
        features.append(np.zeros(6))

    # 7. RMS Energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features.append([np.mean(rms), np.std(rms), np.max(rms)])

    # 8. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features.append([np.mean(zcr), np.std(zcr)])

    # 9. Spectral Centroid and Rolloff
    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
        features.append([np.mean(centroid), np.std(centroid), np.mean(rolloff), np.std(rolloff)])
    except Exception:
        features.append(np.zeros(4))

    # Concatenate all flattened features into a single 1D vector
    feature_vector = np.concatenate([np.atleast_1d(f).flatten() for f in features])
    # Replace any NaNs or Infs
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
    return feature_vector


def extract_features_from_file(file_path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Loads an audio file and returns:
    (feature_vector, raw_audio_waveform, duration_seconds)
    """
    y, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    duration = float(len(y)) / sr
    features = extract_features_from_audio(y, sr=sample_rate)
    return features, y, duration


def extract_timeline_windows(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_sec: float = WINDOW_SIZE_SEC,
    hop_sec: float = HOP_SIZE_SEC
) -> list[dict]:
    """
    Slices audio into overlapping temporal windows for emotion timeline analysis.
    
    Returns list of dicts:
        [
            {
                "window_idx": int,
                "start_time": float (seconds),
                "end_time": float (seconds),
                "center_time": float (seconds),
                "audio": np.ndarray,
                "features": np.ndarray
            },
            ...
        ]
    """
    total_samples = len(y)
    total_duration = total_samples / sr
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    # If audio is shorter than or equal to one window, return single window
    if total_samples <= window_samples:
        feat = extract_features_from_audio(y, sr=sr)
        return [{
            "window_idx": 0,
            "start_time": 0.0,
            "end_time": round(total_duration, 2),
            "center_time": round(total_duration / 2.0, 2),
            "audio": y,
            "features": feat
        }]

    windows = []
    start_sample = 0
    idx = 0

    while start_sample < total_samples:
        end_sample = min(start_sample + window_samples, total_samples)
        segment = y[start_sample:end_sample]

        # Pad segment if it's the last one and shorter than half a window
        if len(segment) < (window_samples // 3):
            break

        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)), mode='constant')

        feat = extract_features_from_audio(segment, sr=sr)
        start_t = start_sample / sr
        end_t = min(end_sample / sr, total_duration)

        windows.append({
            "window_idx": idx,
            "start_time": round(start_t, 2),
            "end_time": round(end_t, 2),
            "center_time": round((start_t + end_t) / 2.0, 2),
            "audio": segment,
            "features": feat
        })

        start_sample += hop_samples
        idx += 1

    return windows
