from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import sys
import cv2
import json
import base64
import numpy as np
from threading import Lock
import time

sys.path.insert(0, '..')
from facialstress import StressNode
from realtimehelper import add_realtime_methods_to_stress_node

# Add real-time processing capability to StressNode
add_realtime_methods_to_stress_node()

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

# Real-time analysis state
realtime_state = {
    'active': False,
    'stress_node': None,
    'results_buffer': [],
    'lock': Lock()
}

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

# ==================== UPLOAD VIDEO ENDPOINT ====================
@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Analyze uploaded video for stress indicators"""
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

# ==================== UPLOAD AUDIO ENDPOINT ====================
@app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio():
    """Analyze uploaded audio for stress indicators"""
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

# ==================== REAL-TIME VIDEO ENDPOINTS ====================
@app.route('/api/realtime/start', methods=['POST'])
def start_realtime():
    """Initialize real-time stress analysis session"""
    try:
        with realtime_state['lock']:
            if realtime_state['active']:
                return jsonify({'error': 'Real-time session already active'}), 400
            
            realtime_state['stress_node'] = StressNode(memory_length=10)
            realtime_state['active'] = True
            realtime_state['results_buffer'] = []
        
        return jsonify({
            'success': True,
            'message': 'Real-time analysis started'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime/stop', methods=['POST'])
def stop_realtime():
    """Stop real-time stress analysis session"""
    try:
        with realtime_state['lock']:
            if not realtime_state['active']:
                return jsonify({'error': 'No active real-time session'}), 400
            
            # Get final summary
            results = realtime_state['results_buffer']
            
            realtime_state['active'] = False
            realtime_state['stress_node'] = None
            realtime_state['results_buffer'] = []
        
        # Calculate summary statistics
        if results:
            avg_stress = sum(r['overall_stress'] for r in results) / len(results)
            max_stress = max(r['overall_stress'] for r in results)
            
            return jsonify({
                'success': True,
                'message': 'Real-time analysis stopped',
                'summary': {
                    'total_frames': len(results),
                    'average_stress': round(avg_stress, 2),
                    'max_stress': round(max_stress, 2),
                    'results': results[-50:]  # Last 50 results
                }
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Real-time analysis stopped',
                'summary': None
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime/frame', methods=['POST'])
def process_realtime_frame():
    """Process a single frame from webcam for stress analysis"""
    try:
        with realtime_state['lock']:
            if not realtime_state['active']:
                return jsonify({'error': 'No active real-time session. Call /api/realtime/start first'}), 400
        
        data = request.get_json()
        
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Decode base64 image
        frame_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        
        # Process frame with stress node
        stress_node = realtime_state['stress_node']
        
        # You'll need to add a method to process single frames
        # This is a simplified version - adapt based on your StressNode implementation
        stress_result = stress_node.process_frame(frame)
        
        result = {
            'timestamp': time.time(),
            'overall_stress': round(stress_result.get('overall_stress', 0.0), 2),
            'classification': stress_result.get('classification', 'Unknown'),
            'emotion_based_stress': round(stress_result.get('emotion_based_stress', 0.0), 2),
            'facial_tension': round(stress_result.get('facial_tension', 0.0), 2),
            'face_detected': stress_result.get('face_detected', False)
        }
        
        # Store in buffer (keep last 100 results)
        with realtime_state['lock']:
            realtime_state['results_buffer'].append(result)
            if len(realtime_state['results_buffer']) > 100:
                realtime_state['results_buffer'].pop(0)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime/status', methods=['GET'])
def realtime_status():
    """Get current real-time analysis status"""
    with realtime_state['lock']:
        return jsonify({
            'active': realtime_state['active'],
            'frames_processed': len(realtime_state['results_buffer'])
        })

# ==================== BATCH ANALYSIS ENDPOINT ====================
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
    print("  POST /api/analyze - Analyze uploaded video")
    print("  POST /api/analyze-audio - Analyze uploaded audio")
    print("  POST /api/analyze-batch - Analyze multiple videos")
    print("\n  Real-time endpoints:")
    print("  POST /api/realtime/start - Start real-time session")
    print("  POST /api/realtime/frame - Process webcam frame")
    print("  POST /api/realtime/stop - Stop real-time session")
    print("  GET  /api/realtime/status - Get session status")
    app.run(debug=True, host='0.0.0.0', port=5000)