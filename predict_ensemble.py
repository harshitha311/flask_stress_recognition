"""
============================================================
  predict_ensemble.py
  ───────────────────
  Aggregate predictions from ALL fold checkpoints for
  more robust inference.

  Two aggregation strategies:
    1. AVERAGE   – mean of probabilities (default)
    2. VOTING    – majority vote on hard labels

  Usage
  ─────
      python predict_ensemble.py --audio path/to/clip.wav
      python predict_ensemble.py --audio path/to/clip.wav --method voting
      python predict_ensemble.py --audio path/to/clip.wav --checkpoints outputs/best_model_fold*.pth

  Output
  ──────
      Prints the ensemble prediction, confidence, and
      per-fold breakdown (optional with --verbose).
============================================================
"""

import argparse, glob, os
import torch
import numpy as np
from feature_extractor import extract_features
from model             import StressNet
from config            import OUTPUT_DIR


def predict_ensemble(
    audio_path: str,
    checkpoint_paths: list[str],
    method: str = "average",
    device: torch.device = None,
) -> dict:
    """
    Load multiple checkpoints, run inference, and aggregate.

    Parameters
    ----------
    audio_path       : str            – path to .wav file
    checkpoint_paths : list[str]      – list of .pth checkpoint paths
    method           : str            – 'average' or 'voting'
    device           : torch.device   – defaults to cuda if available

    Returns
    -------
    dict with keys: label, probability, n_models, per_fold_probs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── extract features once ────────────────────────
    feat = extract_features(audio_path, augment=False)      # (C, T)
    x    = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0) # (1, 1, C, T)
    x    = x.to(device)

    # ── run inference on all checkpoints ─────────────
    n_features     = feat.shape[0]
    all_probs      = []
    all_labels     = []
    fold_breakdown = []

    for ckpt_path in checkpoint_paths:
        model = StressNet(n_features=n_features).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        with torch.no_grad():
            logit = model(x).squeeze().item()
            prob  = 1.0 / (1.0 + np.exp(-logit))   # sigmoid

        all_probs.append(prob)
        all_labels.append(1 if prob >= 0.5 else 0)

        # extract fold ID from filename (e.g., best_model_fold7.pth → 7)
        fold_id = os.path.basename(ckpt_path).replace("best_model_fold", "").replace(".pth", "")
        fold_breakdown.append({
            "fold": fold_id,
            "prob": round(prob, 4),
            "label": "stressed" if prob >= 0.5 else "not stressed",
        })

    # ── aggregate ────────────────────────────────────
    if method == "average":
        final_prob = np.mean(all_probs)
        final_label = "stressed" if final_prob >= 0.5 else "not stressed"
    elif method == "voting":
        # majority vote
        stress_votes = sum(all_labels)
        final_label  = "stressed" if stress_votes > len(all_labels) // 2 else "not stressed"
        # confidence = % of models that agree with the majority
        final_prob   = max(stress_votes, len(all_labels) - stress_votes) / len(all_labels)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'average' or 'voting'.")

    return {
        "label"         : final_label,
        "probability"   : round(final_prob, 4),
        "n_models"      : len(checkpoint_paths),
        "method"        : method,
        "fold_breakdown": fold_breakdown,
    }


# ─── main ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction from multiple fold checkpoints")
    parser.add_argument("--audio",       required=True,  help="Path to .wav audio file")
    parser.add_argument("--checkpoints", nargs="*",      help="Paths to .pth checkpoints (default: all in outputs/)")
    parser.add_argument("--method",      default="average", choices=["average", "voting"],
                        help="Aggregation method: 'average' (mean prob) or 'voting' (majority)")
    parser.add_argument("--verbose",     action="store_true", help="Print per-fold breakdown")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── resolve checkpoints ──────────────────────────
    if args.checkpoints:
        checkpoint_paths = args.checkpoints
    else:
        # auto-discover all best_model_fold*.pth in OUTPUT_DIR
        pattern = os.path.join(OUTPUT_DIR, "best_model_fold*.pth")
        checkpoint_paths = sorted(glob.glob(pattern))

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in '{OUTPUT_DIR}'.\n"
            "Run train.py first, or specify --checkpoints explicitly."
        )

    print(f"[predict_ensemble]  Found {len(checkpoint_paths)} checkpoints")

    # ── run ensemble ─────────────────────────────────
    result = predict_ensemble(args.audio, checkpoint_paths, method=args.method, device=device)

    # ── print results ────────────────────────────────
    print("\n" + "─" * 50)
    print(f"  File        : {args.audio}")
    print(f"  Method      : {result['method']}")
    print(f"  Models used : {result['n_models']}")
    print(f"  Prediction  : {result['label']}")
    print(f"  Confidence  : {result['probability'] * 100:.1f} %")
    print("─" * 50)

    if args.verbose:
        print("\n  Per-fold breakdown:")
        for fold_info in result["fold_breakdown"]:
            print(f"    Fold {fold_info['fold']:>2}:  {fold_info['prob']:.4f}  → {fold_info['label']}")
        print()


if __name__ == "__main__":
    main()
