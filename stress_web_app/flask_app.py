from flask import Flask, request, jsonify, send_from_directory  # ← Added send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import sys
sys.path.insert(0, '..')  # Go up one folder to find stress_recognition.py
from facialstress import StressNode

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')  # ← ADD THIS
def index():  # ← ADD THIS
    """Serve the frontend HTML page"""  # ← ADD THIS
    return send_from_directory('.', 'frontend.html')  # ← ADD THIS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({'status': 'healthy', 'message': 'Stress Recognition API is running'})

# ... rest of your code stays the same
@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Analyze video for stress indicators"""
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run stress analysis
        stress_node = StressNode(memory_length=10)
        avg_stress, off_screen, classification = stress_node.analyze(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Return results
        return jsonify({
            'success': True,
            'results': {
                'overall_stress': round(avg_stress.get('overall_stress', 0.0), 2),
                'classification': classification,
                'emotion_based_stress': round(avg_stress.get('emotion_based_stress', 0.0), 2),
                'facial_tension': round(avg_stress.get('facial_tension', 0.0), 2),
                'stress_variability': round(avg_stress.get('stress_variability', 0.0), 2),
                'off_screen_percentage': round(off_screen * 100, 1)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple videos"""
    try:
        if 'videos' not in request.files:
            return jsonify({'error': 'No video files provided'}), 400
        
        files = request.files.getlist('videos')
        results = []
        
        stress_node = StressNode(memory_length=10)
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Analyze
                avg_stress, off_screen, classification = stress_node.analyze(filepath)
                
                results.append({
                    'filename': filename,
                    'overall_stress': round(avg_stress.get('overall_stress', 0.0), 2),
                    'classification': classification,
                    'emotion_based_stress': round(avg_stress.get('emotion_based_stress', 0.0), 2),
                    'facial_tension': round(avg_stress.get('facial_tension', 0.0), 2),
                    'stress_variability': round(avg_stress.get('stress_variability', 0.0), 2),
                    'off_screen_percentage': round(off_screen * 100, 1)
                })
                
                # Clean up
                os.remove(filepath)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Stress Recognition API...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/analyze - Analyze single video")
    print("  POST /api/analyze-batch - Analyze multiple videos")
    app.run(debug=True, host='0.0.0.0', port=5000)