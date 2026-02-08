import cv2
import numpy as np

def add_realtime_methods_to_stress_node(StressNodeClass=None):
    # Import here to avoid path issues
    if StressNodeClass is None:
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from facialstress import StressNode
        StressNodeClass = StressNode
    
    def process_frame(self, frame):
        try:
            if frame is None or frame.size == 0:
                return {
                    'overall_stress': 0.0,
                    'classification': 'No Frame',
                    'emotion_based_stress': 0.0,
                    'facial_tension': 0.0,
                    'face_detected': False
                }
            
            stress_results = self.recognize_stress(frame)
            
            if not stress_results:
                return {
                    'overall_stress': 0.0,
                    'classification': 'No Face',
                    'emotion_based_stress': 0.0,
                    'facial_tension': 0.0,
                    'face_detected': False
                }
            
            first_face = stress_results[0]
            overall_stress = first_face['stress_level']
            
            if overall_stress < 0.3:
                classification = 'Low Stress'
            elif overall_stress < 0.6:
                classification = 'Moderate Stress'
            else:
                classification = 'High Stress'
            
            return {
                'overall_stress': float(overall_stress),
                'classification': classification,
                'emotion_based_stress': float(first_face['emotion_stress']),
                'facial_tension': float(first_face['facial_tension']),
                'face_detected': True,
                'raw_emotions': first_face['raw_emotions']
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {
                'overall_stress': 0.0,
                'classification': 'Error',
                'emotion_based_stress': 0.0,
                'facial_tension': 0.0,
                'face_detected': False
            }
    
    StressNodeClass.process_frame = process_frame
    print("✓ Added process_frame method to StressNode")
    return StressNodeClass