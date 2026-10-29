"""
Acoustic Audio Generator & Sample Builder
Generates synthetic speech-like acoustic signals with authentic emotion prosody
(pitch contours, formant synthesis, energy envelopes, jitter/shimmer) for both
English and Hindi acoustic profiles.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from src.config import SAMPLE_RATE, EMOTIONS, SAMPLE_AUDIO_DIR

def synthesize_emotional_speech(
    emotion: str,
    language: str = "en",
    duration: float = 3.0,
    sr: int = SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    """
    Synthesizes a rich, multi-formant speech-like audio signal mimicking
    the acoustic prosody, pitch contour, and energy dynamics of a specific emotion.
    
    Acoustic characteristics per emotion:
    - Angry: High F0 (220-350Hz), steep attack, loud bursts, rich high harmonics, harshness
    - Happy: Playful upward-downward melodic pitch sweeps, high brightness, dynamic rhythm
    - Sad: Low F0 (100-140Hz), slow attack, gentle falling pitch contour, subdued energy
    - Fear: Jittery frequency fluctuations, high mean F0, tremolo, breathy aspiration
    - Neutral: Monotone pitch (120-150Hz), steady flat contour, moderate tempo
    - Surprise: Sudden sharp pitch jump at onset, high vocal brightness
    - Disgust: Low F0, elongated vowel drone, guttural subharmonics
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    num_samples = len(t)
    
    # Base language fundamental pitch shift (Hindi vs English intonation nuances)
    lang_f0_offset = 10.0 if language == "hi" else 0.0

    # 1. Pitch (F0) Contour Formulation
    if emotion == "Angry":
        f0_base = 240.0 + lang_f0_offset
        # Aggressive pitch drop followed by forceful rise
        f0_contour = f0_base + 60.0 * np.sin(2 * np.pi * 1.5 * t) + 30.0 * np.cos(2 * np.pi * 3.5 * t)
        amplitude_envelope = (0.7 + 0.3 * np.sin(2 * np.pi * 2.0 * t)) * (0.8 + 0.2 * np.random.uniform(-1, 1, num_samples))
        distortion = 0.25
        breathiness = 0.05
    elif emotion == "Happy":
        f0_base = 210.0 + lang_f0_offset
        # Dynamic melodious rise and fall
        f0_contour = f0_base + 80.0 * np.sin(2 * np.pi * 2.2 * t)
        amplitude_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2
        distortion = 0.05
        breathiness = 0.08
    elif emotion == "Sad":
        f0_base = 115.0 + lang_f0_offset
        # Slow downward dying pitch contour
        f0_contour = f0_base - 25.0 * (t / duration) + 8.0 * np.sin(2 * np.pi * 0.8 * t)
        amplitude_envelope = np.exp(-0.6 * t) * (0.6 + 0.2 * np.sin(2 * np.pi * 1.0 * t))
        distortion = 0.02
        breathiness = 0.15
    elif emotion == "Fear":
        f0_base = 230.0 + lang_f0_offset
        # Rapid jitter and vocal tremor (7-9 Hz vibrato)
        jitter = 35.0 * np.sin(2 * np.pi * 8.0 * t) + 15.0 * np.random.randn(num_samples)
        f0_contour = f0_base + jitter
        amplitude_envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 7.5 * t)
        distortion = 0.08
        breathiness = 0.25
    elif emotion == "Surprise":
        f0_base = 260.0 + lang_f0_offset
        # Sudden pitch spike in first half, tapering down
        pitch_spike = 120.0 * np.exp(-3.0 * (t - 0.3) ** 2)
        f0_contour = f0_base + pitch_spike
        amplitude_envelope = np.exp(-1.5 * t) + 0.4
        distortion = 0.08
        breathiness = 0.12
    elif emotion == "Disgust":
        f0_base = 130.0 + lang_f0_offset
        # Sluggish, low-frequency guttural modulation
        f0_contour = f0_base + 20.0 * np.sin(2 * np.pi * 1.0 * t)
        amplitude_envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 1.5 * t)
        distortion = 0.30  # Vocal fry / guttural sub-harmonics
        breathiness = 0.10
    else:  # Neutral
        f0_base = 140.0 + lang_f0_offset
        # Monotone, steady contour
        f0_contour = f0_base + 10.0 * np.sin(2 * np.pi * 0.5 * t)
        amplitude_envelope = 0.7 * np.ones_like(t)
        distortion = 0.03
        breathiness = 0.05

    # 2. Vocal Formant Synthesis (simulating human vocal tract: F1, F2, F3 resonances)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr
    
    # Glottal pulse simulation
    glottal_source = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.12 * np.sin(4 * phase)
    
    # Formant filters (Vowel /a/ and /e/ formants around 700Hz, 1200Hz, 2400Hz)
    f1_res = np.sin(2 * np.pi * 750 * t) * 0.4
    f2_res = np.sin(2 * np.pi * 1250 * t) * 0.3
    f3_res = np.sin(2 * np.pi * 2500 * t) * 0.2
    
    signal = glottal_source * (1.0 + f1_res + f2_res + f3_res)
    
    # Apply emotion amplitude envelope
    signal = signal * amplitude_envelope
    
    # Add vocal fry / distortion if applicable
    if distortion > 0:
        signal = np.clip(signal * (1.0 + distortion * 2.0), -1.0, 1.0)
    
    # Add breathiness / aspiration noise
    noise = np.random.normal(0, breathiness, num_samples)
    signal = signal + noise
    
    # Smooth start and end with Tukey-like fade
    fade_len = int(0.05 * sr)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out
    
    # Normalize peak amplitude to -1 dB (approx 0.9)
    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal = (signal / max_amp) * 0.90
        
    return signal.astype(np.float32), sr


def generate_sample_audio_files(target_dir: Path = SAMPLE_AUDIO_DIR) -> dict[str, list[str]]:
    """
    Generates standard sample audio files for English and Hindi in all 7 emotions.
    """
    en_dir = target_dir / "english"
    hi_dir = target_dir / "hindi"
    en_dir.mkdir(parents=True, exist_ok=True)
    hi_dir.mkdir(parents=True, exist_ok=True)

    created_files = {"english": [], "hindi": []}

    # English Samples
    for emo in EMOTIONS:
        y, sr = synthesize_emotional_speech(emo, language="en", duration=3.5)
        out_path = en_dir / f"{emo.lower()}_sample.wav"
        sf.write(str(out_path), y, sr)
        created_files["english"].append(str(out_path))

    # Hindi Samples
    hi_names = {
        "Angry": "gussa_angry",
        "Disgust": "ghrina_disgust",
        "Fear": "darr_fear",
        "Happy": "khushi_happy",
        "Neutral": "shant_neutral",
        "Sad": "udasi_sad",
        "Surprise": "aashcharya_surprise"
    }

    for emo in EMOTIONS:
        y, sr = synthesize_emotional_speech(emo, language="hi", duration=3.5)
        hi_name = hi_names[emo]
        out_path = hi_dir / f"{hi_name}_sample.wav"
        sf.write(str(out_path), y, sr)
        created_files["hindi"].append(str(out_path))

    return created_files
