"""
============================================================
  feature_extractor.py
  ────────────────────
  Converts raw .wav audio into fixed-size feature matrices
  that a neural network can consume.

  Features extracted per file
  ───────────────────────────
  1. MFCC matrix          (N_MFCC  × T)   — tonal / spectral shape
  2. Delta MFCCs          (N_MFCC  × T)   — rate-of-change (dynamics)
  3. Delta-delta MFCCs    (N_MFCC  × T)   — acceleration
  4. Mel spectrogram      (N_MELS  × T)   — energy distribution
  5. Chroma STFT          (12      × T)   — pitch-class content
  6. Spectral contrast    (7       × T)   — harmonic vs. noise

  All matrices share the same time-axis T so they can be
  stacked into a single 2-D input image  →  shape (C, T)
  where C = N_MFCC*3 + N_MELS + 12 + 7 = 203 (with defaults).

  Optional augmentations (SpecAugment-style)
  ───────────────────────────────────────────
  • Time-shift  – circular shift of the waveform
  • White noise – additive Gaussian noise
  • Pitch-shift – resample trick (lightweight)
  • Time-mask   – zero out random time frames
  • Freq-mask   – zero out random frequency bands
============================================================
"""

import numpy  as np
import librosa as la
from config import (
    SAMPLE_RATE, DURATION_SEC,
    N_MFCC, N_FFT, HOP_LENGTH, N_MELS,
)

TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)


# ─── waveform helpers ────────────────────────────────────
def _load_and_resample(path: str) -> np.ndarray:
    """Load a .wav at TARGET sample-rate, return 1-D array."""
    y, _ = la.load(path, sr=SAMPLE_RATE, mono=True)
    return y


def _fix_length(y: np.ndarray) -> np.ndarray:
    """Pad with zeros or truncate to exactly TARGET_SAMPLES."""
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]
    return y


# ─── augmentation helpers ───────────────────────────────
def _augment(y: np.ndarray, augment: bool) -> np.ndarray:
    """
    Apply a random subset of augmentations when *augment* is True.
    Each one is applied independently with 50 % probability.
    """
    if not augment:
        return y

    rng = np.random.default_rng()

    # time-shift
    if rng.random() < 0.5:
        shift = rng.integers(-TARGET_SAMPLES // 4, TARGET_SAMPLES // 4)
        y = np.roll(y, shift)

    # white noise (SNR ~ 20 dB)
    if rng.random() < 0.5:
        noise = rng.normal(0, 0.005, size=y.shape).astype(y.dtype)
        y = y + noise

    # pitch-shift  (±2 semitones, lightweight via resampling)
    if rng.random() < 0.5:
        n_steps = rng.uniform(-2, 2)
        y = la.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)
        y = _fix_length(y)                   # pitch-shift can change length

    return y


def _time_mask(spec: np.ndarray, n_masks: int = 2, max_width: int = 10) -> np.ndarray:
    """Zero out *n_masks* random contiguous time-columns."""
    rng  = np.random.default_rng()
    spec = spec.copy()
    T    = spec.shape[1]
    for _ in range(n_masks):
        t0 = rng.integers(0, T)
        t1 = min(t0 + rng.integers(1, max_width + 1), T)
        spec[:, t0:t1] = 0.0
    return spec


def _freq_mask(spec: np.ndarray, n_masks: int = 2, max_width: int = 8) -> np.ndarray:
    """Zero out *n_masks* random contiguous frequency-rows."""
    rng  = np.random.default_rng()
    spec = spec.copy()
    F    = spec.shape[0]
    for _ in range(n_masks):
        f0 = rng.integers(0, F)
        f1 = min(f0 + rng.integers(1, max_width + 1), F)
        spec[f0:f1, :] = 0.0
    return spec


# ─── core feature pipeline ───────────────────────────────
def extract_features(filepath: str, augment: bool = False) -> np.ndarray:
    """
    End-to-end: path  →  normalised 2-D feature matrix  (C × T).

    Parameters
    ----------
    filepath : str   – path to a single .wav file
    augment  : bool  – if True, apply random augmentations (use during training)

    Returns
    -------
    np.ndarray of shape  (C, T),  dtype float32
    """
    # 1. load + resample + fix length
    y = _fix_length(_load_and_resample(filepath))

    # 2. augment the raw waveform
    y = _augment(y, augment)

    # ── spectral features ────────────────────────────────
    # MFCCs + deltas
    mfcc        = la.feature.mfcc(y=y, sr=SAMPLE_RATE,
                                  n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta  = la.feature.delta(mfcc)
    mfcc_delta2 = la.feature.delta(mfcc, order=2)

    # Mel spectrogram (log-scale)
    mel = la.feature.melspectrogram(y=y, sr=SAMPLE_RATE,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel = la.power_to_db(mel, ref=np.max)

    # Chroma
    chroma = la.feature.chroma_stft(y=y, sr=SAMPLE_RATE,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Spectral contrast
    contrast = la.feature.spectral_contrast(y=y, sr=SAMPLE_RATE,
                                            n_fft=N_FFT, hop_length=HOP_LENGTH)

    # ── align time-axis (pad/truncate each to the same T) ─
    T = mfcc.shape[1]                        # reference length
    def _align(m):
        if m.shape[1] < T:
            m = np.pad(m, ((0, 0), (0, T - m.shape[1])))
        return m[:, :T]

    mel      = _align(mel)
    chroma   = _align(chroma)
    contrast = _align(contrast)

    # ── stack into one matrix ────────────────────────────
    # shape → (N_MFCC*3 + N_MELS + 12 + 7,  T)
    feature_matrix = np.vstack([
        mfcc,
        mfcc_delta,
        mfcc_delta2,
        mel,
        chroma,
        contrast,
    ]).astype(np.float32)

    # ── SpecAugment masks (on the stacked spectrogram) ──
    if augment:
        feature_matrix = _time_mask(feature_matrix)
        feature_matrix = _freq_mask(feature_matrix)

    # ── per-feature normalisation (zero mean, unit var) ──
    mu  = feature_matrix.mean(axis=1, keepdims=True)
    std = feature_matrix.std(axis=1, keepdims=True) + 1e-8
    feature_matrix = (feature_matrix - mu) / std

    return feature_matrix


# ── quick smoke-test ─────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage:  python feature_extractor.py <path-to-wav>")
    else:
        feat = extract_features(path, augment=False)
        print(f"Feature matrix shape : {feat.shape}")   # (C, T)
        print(f"Dtype                : {feat.dtype}")
        print(f"Min / Max            : {feat.min():.3f} / {feat.max():.3f}")
