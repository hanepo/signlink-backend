"""
ASL Alphabet ML Detector using trained Decision Tree
"""
import pickle
import numpy as np

class ASLMLDetector:
    """ML-based ASL alphabet detector using trained decision tree"""
    
    def __init__(self, model_path='models/bim_decision_tree.pkl', 
                 labels_path='models/gesture_labels.pkl'):
        """Load trained model and labels"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
            print(f"✅ Loaded ASL model with {len(self.labels)} classes")
        except Exception as e:
            print(f"❌ Failed to load ASL model: {e}")
            self.model = None
            self.labels = []
    
    def extract_features(self, landmarks):
        """
        Extract features from MediaPipe landmarks
        Returns 63-element array (21 landmarks × 3 coords)
        """
        if not landmarks or len(landmarks) != 21:
            return None
        
        features = []
        for lm in landmarks:
            features.extend([lm['x'], lm['y'], lm['z']])
        
        return np.array(features).reshape(1, -1)
    
    def detect_gesture(self, landmarks):
        """
        Detect ASL letter from landmarks
        Returns: letter name (A-Z) or None
        """
        if not self.model or not landmarks:
            return None
        
        features = self.extract_features(landmarks)
        if features is None:
            return None
        
        try:
            prediction = self.model.predict(features)[0]
            # Get confidence using decision tree probability
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(max(probabilities))
            else:
                confidence = 0.85  # Default confidence for non-probabilistic models
            
            return {
                'gesture': prediction,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Detection error: {e}")
            return None
