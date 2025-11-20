# SignLink Gesture Recognition Backend

FastAPI server for real-time BIM (Malaysian Sign Language) gesture detection.

## Features
- MediaPipe Hands landmark extraction
- Rule-based BIM gesture classification (15 gestures)
- RESTful API for mobile app integration
- CORS enabled for cross-origin requests

## Supported Gestures
1. SALAM (Hello)
2. TERIMA KASIH (Thank You)
3. TOLONG (Help)
4. SAYA (I/Me)
5. OK (Good)
6. YA (Yes)
7. TIDAK (No)
8. LIHAT (Look/See)
9. TUNGGU (Wait)
10. PERGI (Go)
11. DATANG (Come)
12. AWAK (You)
13. DIA (He/She)
14. KITA (We)
15. MAAF (Sorry)

## Local Development

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
```

Server will start on `http://localhost:8000`

### Test API
```bash
# Health check
curl http://localhost:8000/

# List gestures
curl http://localhost:8000/gestures

# Classify gesture (requires base64 image)
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"imageBase64": "data:image/jpeg;base64,..."}'
```

## Deployment

### Render
1. Create new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy

### Railway
1. Create new project on Railway
2. Connect GitHub repository
3. Railway auto-detects Python and installs requirements
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy

### Fly.io
1. Install flyctl: `brew install flyctl` (Mac) or download from fly.io
2. Login: `flyctl auth login`
3. Launch app: `flyctl launch`
4. Deploy: `flyctl deploy`

### Environment Variables
- `PORT`: Server port (default: 8000)
- Set by hosting platform automatically

## API Documentation

### Endpoints

#### GET /
Health check and service info
```json
{
  "service": "SignLink Gesture Recognition API",
  "status": "running",
  "version": "1.0.0",
  "gestures_supported": 15
}
```

#### GET /gestures
List all supported gestures with display names and meanings
```json
{
  "gestures": [
    {
      "id": "SALAM",
      "display": "Salam (Hello)",
      "meanings": ["Hello", "Greetings", "Peace"]
    }
  ]
}
```

#### POST /classify
Classify hand gesture from image

**Request:**
```json
{
  "imageBase64": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response (success):**
```json
{
  "gesture": "SALAM",
  "confidence": 0.92,
  "display": "Salam (Hello)",
  "meanings": ["Hello", "Greetings", "Peace"],
  "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0}, ...]
}
```

**Response (no detection):**
```json
{
  "gesture": null,
  "confidence": 0.0,
  "display": null,
  "meanings": [],
  "message": "No hand detected"
}
```

## Performance Tips
- Send images at 0.3-0.5 quality from mobile app
- Limit classification rate to ~1 frame per 1-2 seconds
- Consider resizing images to max 640px width before encoding

## Security Notes
- Images are processed in-memory only (not stored)
- Enable CORS for production (update `allow_origins` in main.py)
- Consider adding API key authentication for production use

## License
MIT
