import os
import torch
import glob
from typing import Tuple
from predict_ensemble import predict_ensemble

class AudioStressDetector:
    def __init__(self, checkpoint_dir: str = None):
        """
        Initialize the audio stress detector with trained models.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # If no checkpoint_dir provided, look in parent directory
        if checkpoint_dir is None:
            # Get the directory where audiostress.py is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_dir = os.path.join(current_dir, "outputs")
        
        self.checkpoint_dir = checkpoint_dir
        
        # Find all checkpoint files
        pattern = os.path.join(checkpoint_dir, "best_model_fold*.pth")
        self.checkpoint_paths = sorted(glob.glob(pattern))
        
        if not self.checkpoint_paths:
            raise FileNotFoundError(
                f"No trained checkpoints found in '{checkpoint_dir}'.\n"
                f"Looking for pattern: {pattern}\n"
                "Make sure you copied the outputs/ directory."
            )
        
        print(f"[AudioStressDetector] Loaded {len(self.checkpoint_paths)} model checkpoints")
    
    def analyze(self, audio_path: str) -> Tuple[float, str, dict]:
        """
        Analyze audio file for stress using ensemble prediction.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (stress_score, classification, details)
        """
        try:
            # Use the predict_ensemble function
            result = predict_ensemble(
                audio_path=audio_path,
                checkpoint_paths=self.checkpoint_paths,
                method="average",
                device=self.device
            )
            
            # Convert binary prediction to stress score
            # result['probability'] is already 0-1 where 1 = stressed
            overall_stress = float(result['probability'])
            
            # Classify based on stress probability
            if overall_stress < 0.3:
                classification = "Low Stress"
            elif overall_stress < 0.7:
                classification = "Moderate Stress"
            else:
                classification = "High Stress"
            
            # Map to emotion label
            if overall_stress >= 0.7:
                detected_emotion = "Highly Stressed"
            elif overall_stress >= 0.5:
                detected_emotion = "Stressed"
            elif overall_stress >= 0.3:
                detected_emotion = "Slightly Stressed"
            else:
                detected_emotion = "Calm"
            
            details = {
                'detected_emotion': detected_emotion,
                'emotion_confidence': float(overall_stress),
                'vocal_stress': float(overall_stress),
                'audio_features': f'Ensemble of {result["n_models"]} models ({result["method"]})',
                'all_emotions': {
                    'stressed': float(overall_stress),
                    'not_stressed': float(1.0 - overall_stress)
                },
                'fold_breakdown': result.get('fold_breakdown', [])
            }
            
            return overall_stress, classification, details
        
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            # Return default values on error
            return 0.5, "Unknown", {
                'detected_emotion': 'Unknown',
                'emotion_confidence': 0.0,
                'vocal_stress': 0.5,
                'audio_features': f'Analysis failed: {str(e)}',
                'all_emotions': {}
            }
    
    def get_stress_classification(self, stress_score: float) -> str:
        """
        Classify stress level based on score.
        
        Args:
            stress_score: Stress score (0-1)
            
        Returns:
            Classification label
        """
        if stress_score < 0.3:
            return "Low Stress"
        elif stress_score < 0.7:
            return "Moderate Stress"
        else:
            return "High Stress"