"""
BIM Gesture Detection using trained model
"""

import pickle
import numpy as np
from pathlib import Path

class BIMGestureDetector:
    def __init__(self, model_path='models/bim_gesture_model.pkl', 
                 labels_path='models/bim_gesture_labels.pkl'):
        """Initialize BIM gesture detector with trained model"""
        model_file = Path(model_path)
        labels_file = Path(labels_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"BIM model not found: {model_path}")
        if not labels_file.exists():
            raise FileNotFoundError(f"BIM labels not found: {labels_path}")
        
        # Load model and labels
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(labels_file, 'rb') as f:
            self.labels = pickle.load(f)
        
        print(f"✅ Loaded BIM model with {len(self.labels)} gestures")
        print(f"   Gestures: {', '.join(sorted(self.labels))}")
    
    def extract_features(self, hand_landmarks):
        """Extract 63 features from MediaPipe hand landmarks"""
        features = []
        for landmark in hand_landmarks:
            features.extend([landmark['x'], landmark['y'], landmark['z']])
        return np.array(features).reshape(1, -1)
    
    def detect_gesture(self, hand_landmarks):
        """
        Detect BIM gesture from hand landmarks
        
        Args:
            hand_landmarks: List of 21 landmarks, each with x, y, z coordinates
                           Format: [{'x': float, 'y': float, 'z': float}, ...]
        
        Returns:
            dict: {'gesture': str, 'confidence': float}
        """
        if not hand_landmarks or len(hand_landmarks) != 21:
            return {'gesture': None, 'confidence': 0.0}
        
        try:
            # Extract features
            features = self.extract_features(hand_landmarks)
            
            # Predict gesture
            prediction = self.model.predict(features)[0]
            
            # Get confidence (probability)
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(max(probabilities))
            
            return {
                'gesture': prediction,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error detecting BIM gesture: {e}")
            return {'gesture': None, 'confidence': 0.0}
    
    def get_supported_gestures(self):
        """Get list of supported BIM gestures"""
        return sorted(self.labels)


# BIM gesture meanings for communication
BIM_GESTURE_MEANINGS = {
    'THUMBS_UP': {'meaning': 'Good/Yes', 'emoji': '👍🏻'},
    'THUMBS_DOWN': {'meaning': 'Bad/No', 'emoji': '👎🏻'},
    'FIST': {'meaning': 'Ready/Wait', 'emoji': '👊🏻'},
    'CLOSED_FIST': {'meaning': 'Stop/Hold', 'emoji': '✊🏻'},
    'FINGERS_CROSSED': {'meaning': 'Hope/Wish', 'emoji': '🤞🏻'},
    'PEACE': {'meaning': 'Peace/Two/Victory', 'emoji': '✌🏻'},
    'SNAP': {'meaning': 'Money/Pay', 'emoji': '🫰🏻'},
    'ILY': {'meaning': 'Love/Care', 'emoji': '🤟🏻'},
    'ROCK': {'meaning': 'Rock/Cool', 'emoji': '🤘🏻'},
    'OK': {'meaning': 'OK/Perfect/Agree', 'emoji': '👌🏻'},
    'PINCH': {'meaning': 'Question/What', 'emoji': '🤌🏻'},
    'SMALL': {'meaning': 'Small/Little/Bit', 'emoji': '🤏🏻'},
    'ONE': {'meaning': 'One/Attention', 'emoji': '☝🏻'},
    'OPEN_HAND': {'meaning': 'Five/Hello', 'emoji': '🖐🏻'},
    'VULCAN': {'meaning': 'Live Long/Prosper', 'emoji': '🖖🏻'},
    'CALL': {'meaning': 'Call/Phone/Shaka', 'emoji': '🤙🏻'},
    'POINT_ME': {'meaning': 'Me/Myself', 'emoji': '👉🏻'},
    'POINT_YOU': {'meaning': 'You/Yourself', 'emoji': '👉🏻'},
}
