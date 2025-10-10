from flask import Flask, request, jsonify, render_template, Response
import cv2
import time
import os
from werkzeug.utils import secure_filename
from camera_predictor import CameraPredictor

app = Flask(__name__)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

# Initialize camera predictor
predictor = CameraPredictor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ Routes ------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "message": "Server is running"
    })

@app.route('/api/status')
def api_status():
    return jsonify({
        "status": "online",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "message": "API is available"
    })

def generate_frames():
    """Generate video frames with predictions"""
    if not predictor.start_camera():
        return
    
    try:
        while predictor.is_running:
            frame = predictor.get_frame_with_prediction()
            if frame is None:
                break
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except GeneratorExit:
        print("Client disconnected from video stream")
    except Exception as e:
        print(f"Camera error: {e}")
    finally:
        predictor.stop_camera()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/prediction')
def get_prediction():
    """Get current prediction"""
    return jsonify(predictor.get_prediction_data())

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera"""
    success = predictor.start_camera()
    return jsonify({"success": success})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera"""
    predictor.stop_camera()
    return jsonify({"success": True})

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """Analyze uploaded video"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Supported: MP4, AVI, MOV, MKV, WMV"}), 400
        
        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded video temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
        
        file.save(temp_path)
        
        # Analyze video
        prediction, confidence = predictor.predict_from_video(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass  # Don't fail if cleanup fails
        
        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "status": "success",
            "message": f"Video analysis completed"
        })
        
    except Exception as e:
        # Clean up temp file in case of error
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
            except:
                pass
        
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ------------------ Error Handlers ------------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large. Maximum size: 100MB"}), 413

# ------------------ Cleanup ------------------

import atexit

def cleanup():
    """Clean up resources on app shutdown"""
    predictor.stop_camera()
    
    # Clean up temp files
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            os.remove(file_path)
        os.rmdir(UPLOAD_FOLDER)
    except:
        pass

atexit.register(cleanup)

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Drishti AI Flask Server")
    print("=" * 60)
    print("\n📡 Available endpoints:")
    print(" 🏠 / - Home page (index.html)")
    print(" 📹 /video_feed - Camera stream with predictions")
    print(" 📊 /api/prediction - Current prediction data")
    print(" 📁 /api/analyze-video - Upload video analysis")
    print(" ❤️ /health - Health check")
    print(f"\n🌐 Server: http://localhost:5000")
    print("=" * 60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        cleanup()
    except Exception as e:
        print(f"\n❌ Server error: {e}")
