"""
SignLink Gesture Recognition API
FastAPI backend for real-time gesture detection (ASL alphabet only)
"""
import base64
import io
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mediapipe as mp
import json
from asl_ml_detector import ASLMLDetector

app = FastAPI(title="SignLink Gesture API", version="3.0.0")

# Enable CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,  # Lower threshold for better detection
    min_tracking_confidence=0.3
)

# Initialize ASL detector
asl_detector = ASLMLDetector()

# ASL alphabet info (A-Z)
ASL_INFO = {}
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    ASL_INFO[letter] = {
        'display': f'Letter {letter}',
        'meanings': [f'Letter {letter}'],
        'category': 'alphabet'
    }
print(f"✅ Loaded {len(ASL_INFO)} ASL letters")


class ImagePayload(BaseModel):
    imageBase64: str


def extract_landmarks(hand_landmarks):
    """Convert MediaPipe landmarks to BIM detector format"""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
    return landmarks


@app.get("/")
def root():
    """API health check"""
    return {
        "service": "SignLink Gesture Recognition API",
        "status": "running",
        "version": "3.0.0",
        "mode": "asl",
        "letters": len(ASL_INFO)
    }


@app.get("/gestures")
def list_gestures():
    """List all supported ASL alphabet letters"""
    return {
        "mode": "asl",
        "gestures": [
            {"id": key, **value}
            for key, value in ASL_INFO.items()
        ]
    }


@app.post("/classify")
def classify_gesture(payload: ImagePayload):
    """Classify ASL alphabet letter from base64 image
    
    Request body:
    {
        "imageBase64": "data:image/jpeg;base64,/9j/4AAQ..."
    }
    
    Response:
    {
        "gesture": "A",
        "confidence": 0.95,
        "display": "Letter A",
        "meanings": ["Letter A"]
    }
    """
    try:
        # Decode base64 image
        image_data = payload.imageBase64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        results = hands.process(img_rgb)

        print(f"[DEBUG] Hand detected: {results.multi_hand_landmarks is not None}")

        if not results.multi_hand_landmarks:
            return {
                "gesture": None,
                "confidence": 0.0,
                "display": None,
                "meanings": [],
                "message": "No hand detected"
            }

        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks_list = extract_landmarks(hand_landmarks)

        # Use ASL detector
        print(f"[DEBUG] Using ASL detector with {len(landmarks_list)} landmarks")
        result = asl_detector.detect_gesture(landmarks_list)
        print(f"[DEBUG] ASL result: {result}")
        if result:
            gesture = result['gesture']
            confidence = result['confidence']
            info = ASL_INFO.get(gesture, {
                'display': f'Letter {gesture}',
                'meanings': [f'Letter {gesture}'],
                'category': 'alphabet'
            })
            return {
                "gesture": gesture,
                "confidence": confidence,
                "display": info['display'],
                "meanings": info['meanings'],
                "category": info.get('category', 'alphabet'),
                "mode": "asl"
            }

        return {
            "gesture": None,
            "confidence": 0.0,
            "display": None,
            "meanings": [],
            "message": "No ASL letter recognized",
            "mode": "asl"
        }
            
    except Exception as e:
        print(f"Classification error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint for deployment platforms"""
    return {"status": "healthy"}
