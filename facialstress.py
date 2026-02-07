"""
This file defines the `FaceNode` class, which is used for facial emotion recognition.
"""

import os
from fer import FER
import torch
import cv2
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import argparse
from termcolor import colored, cprint
import time
import numpy as np


class StressNode:
    """
    A class for detecting faces and recognizing stress indicators in images.
    This node analyzes facial expressions, micro-expressions, and behavioral patterns
    to estimate stress levels over a sequence of frames.
    """

    def __init__(self, memory_length: int = 10):
        """
        Initializes the StressNode with a specified memory length for storing detected stress indicators.

        Args:
            memory_length (int): Number of frames of stress data to remember.
        """
        self.detector = FER()  # Use for facial expression analysis
        self.memory_length = memory_length
        self.stress_data: List[Dict[str, float]] = []  # List to store stress indicators per frame
        
        # Stress-related emotion weights (higher values indicate more stress)
        self.stress_emotion_weights = {
            'angry': 0.8,
            'disgust': 0.6,
            'fear': 0.9,
            'sad': 0.7,
            'surprise': 0.5,
            'neutral': 0.0,
            'happy': -0.3  # Negative correlation with stress
        }

    def calculate_stress_from_emotions(self, emotions: Dict[str, float]) -> float:
        """
        Calculates a stress score based on facial emotion distribution.

        Args:
            emotions (Dict[str, float]): Dictionary of emotion scores.

        Returns:
            float: Weighted stress score (0-1 range).
        """
        stress_score = 0.0
        total_weight = 0.0
        
        for emotion, score in emotions.items():
            if emotion in self.stress_emotion_weights:
                weight = self.stress_emotion_weights[emotion]
                stress_score += score * weight
                total_weight += abs(weight) * score
        
        # Normalize to 0-1 range
        if total_weight > 0:
            stress_score = max(0, min(1, (stress_score + 0.3) / 1.2))
        
        return stress_score

    def detect_face_tension(self, face_box: Tuple[int, int, int, int], frame: cv2.Mat) -> float:
        """
        Analyzes facial region for tension indicators (simplified version).
        In a production system, this would analyze eye openness, jaw tension, etc.

        Args:
            face_box (Tuple[int, int, int, int]): Bounding box coordinates (x, y, w, h).
            frame (cv2.Mat): The image frame.

        Returns:
            float: Tension score (0-1 range).
        """
        try:
            x, y, w, h = face_box
            face_roi = frame[y:y+h, x:x+w]
            
            # Simple edge detection as proxy for facial tension
            # (More sophisticated methods would analyze specific facial landmarks)
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Normalize edge density to approximate tension
            tension_score = min(1.0, edge_density * 5)
            
            return tension_score
        except Exception as e:
            return 0.0

    def recognize_stress(self, frame: cv2.Mat) -> List[Dict[str, float]]:
        """
        Analyzes an image to identify faces and determine stress indicators.

        Args:
            frame (cv2.Mat): Image to analyze for stress.

        Returns:
            List[Dict[str, float]]: A list of dictionaries containing stress scores for each detected face.
        """
        if frame is None or frame.size == 0:
            return []

        # Detect faces and their emotions
        faces = self.detector.detect_emotions(frame)
        
        stress_results = []
        
        for face in faces:
            emotions = face['emotions']
            face_box = face['box']
            
            # Calculate stress indicators
            emotion_stress = self.calculate_stress_from_emotions(emotions)
            tension_score = self.detect_face_tension(face_box, frame)
            
            # Combine indicators (weighted average)
            overall_stress = (emotion_stress * 0.7) + (tension_score * 0.3)
            
            stress_result = {
                'box': face_box,
                'stress_level': overall_stress,
                'emotion_stress': emotion_stress,
                'facial_tension': tension_score,
                'raw_emotions': emotions
            }
            
            stress_results.append(stress_result)
        
        # Store in memory
        if stress_results:
            frame_stress = {
                'faces': stress_results,
                'avg_stress': np.mean([f['stress_level'] for f in stress_results])
            }
        else:
            frame_stress = {
                'faces': [],
                'avg_stress': 0.0
            }
        
        self.stress_data.append(frame_stress)
        if len(self.stress_data) > self.memory_length:
            self.stress_data.pop(0)
        
        return stress_results

    def get_avg_stress(self) -> Dict[str, float]:
        """
        Computes the average stress indicators over the stored frames in the node's memory.

        Returns:
            Dict[str, float]: A dictionary containing average stress metrics.
        """
        if not self.stress_data:
            return {
                'overall_stress': 0.0,
                'emotion_based_stress': 0.0,
                'facial_tension': 0.0,
                'stress_variability': 0.0
            }
        
        all_stress_levels = []
        all_emotion_stress = []
        all_tension = []
        
        for frame_data in self.stress_data:
            for face in frame_data['faces']:
                all_stress_levels.append(face['stress_level'])
                all_emotion_stress.append(face['emotion_stress'])
                all_tension.append(face['facial_tension'])
        
        if not all_stress_levels:
            return {
                'overall_stress': 0.0,
                'emotion_based_stress': 0.0,
                'facial_tension': 0.0,
                'stress_variability': 0.0
            }
        
        return {
            'overall_stress': float(np.mean(all_stress_levels)),
            'emotion_based_stress': float(np.mean(all_emotion_stress)),
            'facial_tension': float(np.mean(all_tension)),
            'stress_variability': float(np.std(all_stress_levels))
        }

    def get_stress_classification(self, stress_score: float) -> str:
        """
        Classifies stress level based on numerical score.

        Args:
            stress_score (float): Stress score (0-1 range).

        Returns:
            str: Classification label.
        """
        if stress_score < 0.3:
            return "Low Stress"
        elif stress_score < 0.6:
            return "Moderate Stress"
        else:
            return "High Stress"

    def clear_memory(self) -> None:
        """
        Clears the node's memory of detected stress data.
        """
        self.stress_data = []

    def analyze(self, video_path: str) -> Tuple[Dict[str, float], float, str]:
        """
        Analyzes a video for stress indicators and calculates the percentage of frames with no detected faces.

        Args:
            video_path (str): Path to the video file.

        Returns:
            Tuple[Dict[str, float], float, str]: Average stress metrics, percentage of frames without faces,
                                                  and overall stress classification.
        """
        self.clear_memory()

        video = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not video.isOpened():
            print("Error: Could not open video.")
            return {}, 0.0, "Unknown"

        frames = []
        while True:
            # Read the next frame from the video
            ret, frame = video.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                break

            # Add the frame to the list
            frames.append(frame)

        # Release the video capture object
        video.release()

        empty_frames = 0
        
        for frame in frames:
            try:
                stress_results = self.recognize_stress(frame)
                if not stress_results:  # Check if no faces were detected in the frame
                    empty_frames += 1
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
        
        avg_stress = self.get_avg_stress()
        off_screen_percent = empty_frames / len(frames) if frames else 0
        stress_classification = self.get_stress_classification(avg_stress.get('overall_stress', 0.0))
        
        return avg_stress, off_screen_percent, stress_classification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stress Recognition Node")
    parser.add_argument('path', type=str, help='Path to the video file.')
    args = parser.parse_args()

    cprint('Initializing Stress Recognition Node...', 'green', attrs=['bold'])
    node = StressNode()

    cprint('Analyzing Data...', 'green', attrs=['bold'])
    avg_stress, off_screen, classification = node.analyze(args.path)
    
    print("\n" + "="*50)
    cprint("STRESS ANALYSIS RESULTS", 'cyan', attrs=['bold'])
    print("="*50)
    print(f"\nOverall Stress Level: {avg_stress.get('overall_stress', 0.0):.2f}")
    print(f"Classification: {classification}")
    print(f"\nEmotion-Based Stress: {avg_stress.get('emotion_based_stress', 0.0):.2f}")
    print(f"Facial Tension: {avg_stress.get('facial_tension', 0.0):.2f}")
    print(f"Stress Variability: {avg_stress.get('stress_variability', 0.0):.2f}")
    print(f"\nOff-Screen Percentage: {off_screen*100:.1f}%")
    print("="*50 + "\n")