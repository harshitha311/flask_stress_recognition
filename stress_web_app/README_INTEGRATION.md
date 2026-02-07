# Stress Recognition System - Frontend Integration Guide

## 🚀 Quick Start

### Step 1: Install Backend Dependencies
```bash
pip install flask flask-cors
```

### Step 2: Start the Backend API
```bash
# Navigate to your project folder
cd C:\Users\dell\Desktop\flask_stress_detection\stress_web_app

# Run the Flask server
python flask_app.py
```

You should see:
```
Starting Stress Recognition API...
API will be available at: http://localhost:5000
```

### Step 3: Open the Frontend

Simply open `frontend.html` in your web browser (double-click the file or drag it into Chrome/Firefox/Edge).

### Step 4: Test It!

1. Click the upload area or drag a video file
2. Click "Analyze Video"
3. Wait for results to appear

---

## 📁 File Structure
```
stress_detection_model/
├── facialstress.py     # Your stress recognition code
└── stress_web_app/
    ├── flask_app.py          # Backend API server
    ├── frontend.html         # Frontend UI
    ├── requirements_backend.txt
    └── README_INTEGRATION.md
```

---

## 🔌 API Endpoints

### Health Check
```bash
GET http://localhost:5000/api/health
```

### Analyze Single Video
```bash
POST http://localhost:5000/api/analyze
Content-Type: multipart/form-data
Body: video file
```

Response:
```json
{
  "success": true,
  "results": {
    "overall_stress": 0.45,
    "classification": "Moderate Stress",
    "emotion_based_stress": 0.52,
    "facial_tension": 0.28,
    "stress_variability": 0.15,
    "off_screen_percentage": 12.3
  }
}
```

### Analyze Multiple Videos (Batch)
```bash
POST http://localhost:5000/api/analyze-batch
Content-Type: multipart/form-data
Body: multiple video files with key "videos"
```

---

## 🧪 Testing the API with cURL
```bash
# Health check
curl http://localhost:5000/api/health

# Analyze video
curl -X POST -F "video=@test_video.mp4" http://localhost:5000/api/analyze
```

---

## 🐛 Troubleshooting

### CORS Error
Make sure you installed `flask-cors`:
```bash
pip install flask-cors
```

### Import Error
Make sure `stress_recognition.py` is in the parent folder of `stress_web_app`:
```
stress_detection_model/
├── stress_recognition.py  ← Should be here
└── stress_web_app/
    └── flask_app.py
```

### Port Already in Use
Change the port in `flask_app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Changed to 5001
```

And update the frontend URL in `frontend.html`:
```javascript
const response = await fetch('http://localhost:5001/api/analyze', {
```

### Module Not Found Error
Make sure all dependencies are installed:
```bash
pip install -r requirements_backend.txt
```

---

## 📱 Features

✅ Drag & drop video upload  
✅ Beautiful gradient UI  
✅ Real-time analysis  
✅ Color-coded stress levels (Green/Yellow/Red)  
✅ Detailed metrics display  
✅ Mobile responsive  
✅ Batch video analysis support  

---

## 🎨 Customization

### Change Colors
Edit the CSS in `frontend.html`:
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Change Port
Edit `flask_app.py` (last line):
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change 5000 to your port
```

---

## 📊 Next Steps

1. Deploy to cloud (Heroku, AWS, Google Cloud)
2. Add user authentication
3. Store results in database
4. Create dashboard with charts
5. Add email notifications
6. Real-time webcam analysis

---

## ✨ How It Works

1. User uploads video through web interface
2. Frontend sends video to Flask API (`http://localhost:5000/api/analyze`)
3. API saves the video temporarily
4. API calls `StressNode.analyze()` from your `stress_recognition.py`
5. Results returned as JSON
6. Frontend displays results with beautiful UI and color coding

---

## 🚀 Ready to Deploy?

For production deployment:
1. Set `debug=False` in `flask_app.py`
2. Use a production server like Gunicorn
3. Add authentication
4. Use HTTPS
5. Add rate limiting

---

That's it! You now have a fully functional web-based stress recognition system! 