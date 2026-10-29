"""
Emotion Intensity Calculator
Analyzes vocal energy, pitch dynamics (F0), spectral tilt, and activation to quantify
speech emotion intensity on a 0.0 to 1.0 (0-100%) scale with bilingual descriptors.
"""

import numpy as np
import librosa
from src.config import INTENSITY_LEVELS, SAMPLE_RATE

def calculate_emotion_intensity(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Computes multifaceted acoustic emotion intensity from an audio waveform.
    
    Returns:
        dict containing:
            - intensity_score (0.0 to 1.0)
            - intensity_percentage (0 - 100%)
            - level_en ("Subtle", "Moderate", "High", "Intense")
            - level_hi ("हल्का", "मध्यम", "तीव्र", "अत्यधिक तीव्र")
            - arousal_score (0.0 to 1.0)
            - pitch_variability (Hz)
            - energy_rms (float)
    """
    if len(y) == 0:
        return {
            "intensity_score": 0.0,
            "intensity_percentage": 0.0,
            "level_en": "Subtle",
            "level_hi": "हल्का",
            "arousal_score": 0.0,
            "pitch_variability": 0.0,
            "energy_rms": 0.0
        }
    
    # 1. Root Mean Square (RMS) Energy (Acoustic Loudness & Dynamics)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms)) if len(rms) > 0 else 0.0
    rms_std = float(np.std(rms))
    
    # Normalize RMS to [0, 1] range based on typical speech dB
    norm_rms = float(np.clip(rms_mean * 12.0, 0.0, 1.0))
    norm_rms_dyn = float(np.clip((rms_max - rms_mean) * 10.0, 0.0, 1.0))
    
    # 2. Pitch / Fundamental Frequency (F0) Dynamics
    try:
        # Fast autocorrelation-based or piptrack pitch estimation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=60, fmax=450)
        pitch_vals = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 60:
                pitch_vals.append(pitch)
        
        if len(pitch_vals) > 5:
            pitch_std = float(np.std(pitch_vals))
            pitch_mean = float(np.mean(pitch_vals))
            # Standard vocal pitch variability normalized: typical expressive speech varies 20-80 Hz
            norm_pitch_var = float(np.clip(pitch_std / 70.0, 0.0, 1.0))
        else:
            pitch_std = 0.0
            pitch_mean = 150.0
            norm_pitch_var = 0.3
    except Exception:
        pitch_std = 0.0
        pitch_mean = 150.0
        norm_pitch_var = 0.3

    # 3. Spectral Centroid (Vocal Brightness / Cord Tension)
    try:
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        cent_mean = float(np.mean(spec_cent))
        # Speech centroids typically range from 1000 Hz (subdued) to 3500+ Hz (tense/screaming)
        norm_centroid = float(np.clip((cent_mean - 1000.0) / 2500.0, 0.0, 1.0))
    except Exception:
        norm_centroid = 0.4

    # 4. Zero Crossing Rate (Voice Shrillness / Breathiness)
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))
        norm_zcr = float(np.clip(zcr_mean * 8.0, 0.0, 1.0))
    except Exception:
        norm_zcr = 0.3

    # Composite Emotional Arousal & Intensity Formula:
    # 40% Vocal Energy + 30% Pitch Excursion + 20% Spectral Brightness + 10% Dynamic Contrast
    composite_intensity = (
        0.40 * norm_rms +
        0.30 * norm_pitch_var +
        0.20 * norm_centroid +
        0.10 * norm_rms_dyn
    )
    
    # Sigmoidal calibration to give natural human perception scaling
    calibrated_intensity = float(np.clip(1.0 / (1.0 + np.exp(-6.0 * (composite_intensity - 0.45))), 0.05, 0.98))
    
    # Categorize into intensity level
    level_en = "Moderate"
    level_hi = "मध्यम"
    for low, high, en_label, hi_label in INTENSITY_LEVELS:
        if low <= calibrated_intensity <= high:
            level_en = en_label
            level_hi = hi_label
            break

    return {
        "intensity_score": round(calibrated_intensity, 3),
        "intensity_percentage": round(calibrated_intensity * 100, 1),
        "level_en": level_en,
        "level_hi": level_hi,
        "arousal_score": round(float(np.clip(composite_intensity, 0.0, 1.0)), 3),
        "pitch_variability_hz": round(pitch_std, 1),
        "energy_rms": round(rms_mean, 4)
    }
