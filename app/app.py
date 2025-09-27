from flask import Flask, request, jsonify, render_template, Response
import cv2
import os
import time
import atexit

app = Flask(__name__)

# ------------------ Global Variables ------------------
camera_manager = None
latest_prediction = {"prediction": "No AI model connected", "confidence": 0.0}

# ------------------ Camera Manager ------------------
class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_active = False
    
    def start_camera(self):
        if not self.is_active:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.is_active = True
                return True
            else:
                print("Failed to open camera")
                return False
        return True
    
    def read_frame(self):
        if self.cap and self.is_active:
            return self.cap.read()
        return False, None
    
    def release_camera(self):
        if self.cap:
            self.cap.release()
            self.is_active = False
            print("Camera released")

camera_manager = CameraManager()

# ------------------ Video Streaming ------------------
def gen_frames():
    """Generate frames for video streaming (no AI overlay)"""
    if not camera_manager.start_camera():
        return
    
    try:
        while True:
            success, frame = camera_manager.read_frame()
            if not success:
                break
            
            # Header overlay
            header_height = 80
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_height), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Title text
            title_text = "Drishti AI - Camera Active (No AI Model)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(title_text, font, 0.7, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, title_text, (text_x, 30), font, 0.7, (255, 255, 255), 2)
            
            # Timestamp
            time_text = f"Time: {time.strftime('%H:%M:%S')}"
            cv2.putText(frame, time_text, (20, 65), font, 0.5, (0, 255, 0), 1)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    except Exception as e:
        print(f"Error in gen_frames: {e}")
    finally:
        camera_manager.release_camera()

# ------------------ Flask Routes ------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Mock prediction endpoint"""
    if request.method == 'GET':
        return jsonify(latest_prediction)
    return jsonify({
        'prediction': 'No AI Model Available',
        'confidence': 0.0,
        'error': 'AI models removed in this version'
    })

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Mock base64 prediction endpoint"""
    return jsonify({
        'prediction': 'No AI Model Available',
        'confidence': 0.0,
        'error': 'AI models removed in this version'
    })

@app.route('/camera/status')
def camera_status():
    return jsonify({
        'is_active': camera_manager.is_active,
        'status': 'active' if camera_manager.is_active else 'inactive'
    })

@app.route('/camera/restart', methods=['POST'])
def camera_restart():
    try:
        camera_manager.release_camera()
        success = camera_manager.start_camera()
        return jsonify({
            'success': success,
            'message': 'Camera restarted successfully' if success else 'Failed to restart camera'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error restarting camera: {str(e)}"
        })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'camera_available': camera_manager is not None,
        'ai_models_loaded': False,
        'message': 'Flask server running without AI models'
    })

@app.route('/landmarks', methods=['POST'])
def get_landmarks():
    return jsonify({
        'error': 'Landmark extraction not available - AI models removed',
        'landmarks': [],
        'count': 0
    })

# ------------------ Cleanup ------------------
def cleanup():
    if camera_manager:
        camera_manager.release_camera()

atexit.register(cleanup)

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Drishti AI Flask Server (Simplified Version)")
    print("=" * 50)
    print("✓ Camera functionality: Available")
    print("✗ AI Prediction functionality: Disabled")
    print("✓ Web Interface: Available")
    print("✓ Server running on: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
