from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import sys
sys.path.insert(0, '..')
from facialstress import StressNode

# Add debug print here
print("DEBUG: Attempting to import AudioStressDetector...")
try:
    from audiostress import AudioStressDetector
    print("DEBUG: AudioStressDetector imported successfully!")
except Exception as e:
    print(f"DEBUG: Failed to import AudioStressDetector: {e}")
    import traceback
    traceback.print_exc()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

def allowed_file(filename, file_type='video'):
    if file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    else:  # audio
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

@app.route('/')
def index():
    """Serve the frontend HTML page"""
    return send_from_directory('.', 'frontend.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({'status': 'healthy', 'message': 'Stress Recognition API is running'})

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Analyze video for stress indicators"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, 'video'):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run stress analysis
        stress_node = StressNode(memory_length=10)
        avg_stress, off_screen, classification = stress_node.analyze(filepath)
        
        # Clean up
        os.remove(filepath)
        
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

@app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio():
    """Analyze audio for stress indicators"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, 'audio'):
            return jsonify({'error': 'Invalid file type. Allowed: wav, mp3, m4a'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run audio stress analysis
        audio_detector = AudioStressDetector()
        overall_stress, classification, details = audio_detector.analyze(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
        'success': True,
        'results': {
        'overall_stress': float(round(overall_stress, 2)),
        'classification': classification,
        'vocal_stress': float(details.get('vocal_stress', 0.0)),
        'audio_features': details.get('audio_features', 'N/A'),
        'detected_emotion': details.get('detected_emotion', 'Unknown'),
        'emotion_confidence': float(round(details.get('emotion_confidence', 0.0), 2)),
        'all_emotions': {k: float(v) for k, v in details.get('all_emotions', {}).items()}
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
            if file and allowed_file(file.filename, 'video'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
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
    print("  POST /api/analyze - Analyze video")
    print("  POST /api/analyze-audio - Analyze audio")
    print("  POST /api/analyze-batch - Analyze multiple videos")
    app.run(debug=True, host='0.0.0.0', port=5000)