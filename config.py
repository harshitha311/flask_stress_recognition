"""
============================================================
  RAVDESS Audio Stress Detection — Configuration
============================================================
  Centralised hyper-parameters & paths.
  Edit DATASET_ROOT to point to your downloaded folder.
============================================================
"""

import os

# ── Paths ────────────────────────────────────────────────
# After downloading Audio_Speech_Actors_01-24.zip from Zenodo,
# extract it so the layout is:
#   DATASET_ROOT/
#     Actor_01/
#       03-01-01-01-01-01-01.wav
#       ...
#     Actor_02/
#     ...
DATASET_ROOT = os.path.expanduser("dataset")   # <── change this

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── RAVDESS filename codec ───────────────────────────────
# position → meaning
# 0  modality        03 = audio-only
# 1  vocal-channel   01 = speech | 02 = song
# 2  emotion         01-08  (see EMOTION_MAP)
# 3  intensity       01 = normal | 02 = strong
# 4  statement       01 | 02
# 5  repetition      01 | 02
# 6  actor           01-24  (odd=male, even=female)

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# ── Stress-label mapping ─────────────────────────────────
# Stress-indicative emotions → 1   |   Non-stress → 0
# Rationale: angry, fearful, disgust are high-arousal negative
#            states strongly correlated with physiological stress.
#            surprised is excluded because it is ambiguous
#            (can be positive or negative).
STRESS_EMOTIONS     = {"angry", "fearful", "disgust"}
NON_STRESS_EMOTIONS = {"neutral", "calm", "happy", "sad", "surprised"}

# ── Audio preprocessing ─────────────────────────────────
SAMPLE_RATE  = 16_000   # resample everything to 16 kHz
DURATION_SEC = 3.0      # pad / truncate to 3 s
N_MFCC       = 40       # number of MFCC coefficients
N_FFT        = 512
HOP_LENGTH   = 160      # 10 ms @ 16 kHz
N_MELS       = 64

# ── Train / eval ─────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42
EPOCHS       = 50
BATCH_SIZE   = 32
LEARNING_RATE = 1e-3
