from flask import Flask, request, jsonify, render_template, Response
import pickle
import numpy as np
import cv2
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image
import os
import sys
import xgboost as xgb

# Add src directory to path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.json')
model = xgb.Booster()
model.load_model(MODEL_PATH)

# Define sign classes
SIGN_CLASSES = ["Horse", "Mouse", "Dress", "Hat", "Shirt", "Soap", "Table", "Chair", "Door", "Fan", 
                "Window", "TV", "Clock", "Bed", "Car", "Book", "Pen", "Phone", "Laptop", "Water",
                "Food", "Tea", "Coffee", "Milk", "Juice", "Good", "Bad", "Hello", "Thank You"]

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Store latest prediction globally (for /predict endpoint)
latest_prediction = {"prediction": "N/A", "confidence": 0.0}

# Global camera instance to prevent multiple access conflicts
camera = None

class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_active = False
    
    def start_camera(self):
        if not self.is_active:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                # Set camera properties for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

# Initialize global camera manager
camera_manager = CameraManager()

def extract_landmarks(image):
    """Extract holistic landmarks from image"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_image)
    
    landmarks = []
    
    # Extract face landmarks if available (468 points × 3 coordinates)
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0.0] * (468 * 3))  # Pad with zeros if no face detected
                
    # Extract pose landmarks if available (33 points × 4 coordinates)
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    else:
        landmarks.extend([0.0] * (33 * 4))  # Pad with zeros if no pose detected
                
    # Extract left hand landmarks if available (21 points × 3 coordinates)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0.0] * (21 * 3))  # Pad with zeros
                
    # Extract right hand landmarks if available (21 points × 3 coordinates)
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0.0] * (21 * 3))  # Pad with zeros
    
    # Convert to numpy array and ensure proper shape
    landmarks_array = np.array(landmarks, dtype=np.float32).reshape(1, -1)
    
    # Return landmarks if we have any detection, otherwise None
    if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks or results.face_landmarks:
        return landmarks_array, results
    return None, None

def decode_base64_image(base64_string):
    """Decode base64 image string to OpenCV format"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def gen_frames():
    """Generate webcam frames with predictions"""
    global latest_prediction, camera_manager
    
    # Use the global camera manager to prevent conflicts
    if not camera_manager.start_camera():
        return
    
    try:
        while True:
            success, frame = camera_manager.read_frame()
            if not success:
                print("Failed to read frame from camera")
                break

            # Extract landmarks
            landmarks, results = extract_landmarks(frame)
            
            # Always show status on frame
            status_text = "No pose detected"
            status_color = (0, 0, 255)  # Red (BGR format)
            
            if landmarks is not None and results is not None:
                try:
                    # Draw face landmarks with elegant style (reduced for performance)
                    if results.face_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1))
                    
                    # Draw pose landmarks with neon effect
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1))
                    
                    # Draw left hand landmarks with bright colors
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2))
                    
                    # Draw right hand landmarks with bright colors
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(254, 44, 85), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(255, 84, 255), thickness=2))
                    
                    # Use XGBoost model for prediction
                    dmatrix = xgb.DMatrix(landmarks)
                    raw_prediction = model.predict(dmatrix)
                    
                    # Handle the prediction output with better error handling
                    predicted_sign = "Unknown"
                    confidence = 0.0
                    pred_class_idx = 0
                    
                    try:
                        if isinstance(raw_prediction, np.ndarray) and raw_prediction.size > 0:
                            if raw_prediction.size > 1:
                                # Multi-class case
                                pred_class_idx = int(np.argmax(raw_prediction))
                                confidence = float(np.max(raw_prediction))
                                predicted_sign = SIGN_CLASSES[pred_class_idx] if pred_class_idx < len(SIGN_CLASSES) else "Unknown"
                            else:
                                # Binary case
                                pred_value = float(raw_prediction[0])
                                pred_class_idx = int(pred_value > 0.5)
                                confidence = pred_value if pred_value > 0.5 else (1.0 - pred_value)
                                predicted_sign = SIGN_CLASSES[pred_class_idx] if pred_class_idx < len(SIGN_CLASSES) else ("Good Posture" if pred_class_idx == 1 else "Bad Posture")
                        else:
                            # Scalar prediction
                            pred_value = float(raw_prediction)
                            pred_class_idx = int(pred_value > 0.5)
                            confidence = pred_value if pred_value > 0.5 else (1.0 - pred_value)
                            predicted_sign = SIGN_CLASSES[pred_class_idx] if pred_class_idx < len(SIGN_CLASSES) else ("Good Posture" if pred_class_idx == 1 else "Bad Posture")
                    except (IndexError, ValueError) as e:
                        print(f"Prediction processing error: {e}")
                        predicted_sign = "Error"
                        confidence = 0.0

                    latest_prediction = {
                        "prediction": predicted_sign,
                        "confidence": min(max(confidence, 0.0), 1.0),  # Clamp between 0-1
                        "class_index": int(pred_class_idx)
                    }

                    # Update status for display
                    status_text = f"Sign: {predicted_sign} ({confidence:.2f})"
                    status_color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    status_text = f"Error: {str(e)[:30]}"
                    status_color = (0, 165, 255)  # Orange
            
            # Add stylish header with improved design
            header_height = 80
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_height), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Add title with better styling
            title_text = "Drishti AI - Sign Language Detection"
            font_scale = 0.8
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Calculate text size for centering
            text_size = cv2.getTextSize(title_text, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            
            cv2.putText(frame, title_text, (text_x, 35), font, font_scale, (255, 255, 255), thickness)
            
            # Add status with better positioning
            status_y = header_height - 15
            cv2.putText(frame, status_text, (20, status_y), font, 0.6, status_color, 2)

            # Encode frame for streaming with better compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    except GeneratorExit:
        print("Client disconnected")
    except Exception as e:
        print(f"Camera error: {e}")
    finally:
        print("Frame generation ended")

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Webcam video streaming route"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return latest prediction (for frontend polling) OR handle uploaded image"""
    global latest_prediction

    if request.method == 'GET':
        return jsonify(latest_prediction)

    # POST image prediction (original logic)
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Read and process image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        landmarks, _ = extract_landmarks(image)

        if landmarks is None:
            return jsonify({'error': 'No pose detected in image'}), 400

        # Use XGBoost model for prediction
        dmatrix = xgb.DMatrix(landmarks)
        raw_prediction = model.predict(dmatrix)

        # Handle model prediction based on output shape
        if isinstance(raw_prediction, np.ndarray):
            if raw_prediction.size > 1:
                pred_class_idx = int(np.argmax(raw_prediction))
                confidence = float(raw_prediction[pred_class_idx])
                predicted_class = SIGN_CLASSES[pred_class_idx] if pred_class_idx < len(SIGN_CLASSES) else "Unknown"
            else:
                pred_value = float(raw_prediction[0])
                predicted_class = "Good Posture" if pred_value > 0.5 else "Bad Posture"
                confidence = pred_value if pred_value > 0.5 else (1.0 - pred_value)
        else:
            pred_value = float(raw_prediction)
            predicted_class = "Good Posture" if pred_value > 0.5 else "Bad Posture"
            confidence = pred_value if pred_value > 0.5 else (1.0 - pred_value)
        
        result = {
            'prediction': predicted_class,
            'confidence': float(confidence)
        }

        latest_prediction = result
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Predict pose classification from base64 encoded image"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image = decode_base64_image(data['image'])
        landmarks, _ = extract_landmarks(image)

        if landmarks is None:
            return jsonify({'error': 'No pose detected in image'}), 400

        # Use XGBoost model for prediction
        dmatrix = xgb.DMatrix(landmarks)
        raw_prediction = model.predict(dmatrix)
        
        # Handle prediction
        if isinstance(raw_prediction, np.ndarray):
            if raw_prediction.size > 1:
                pred_class_idx = int(np.argmax(raw_prediction))
                confidence = float(raw_prediction[pred_class_idx])
                predicted_class = SIGN_CLASSES[pred_class_idx] if pred_class_idx < len(SIGN_CLASSES) else "Unknown"
            else:
                pred_value = float(raw_prediction[0])
                predicted_class = "Good Posture" if pred_value > 0.5 else "Bad Posture"
                confidence = pred_value if pred_value > 0.5 else (1.0 - pred_value)
        else:
            pred_value = float(raw_prediction)
            predicted_class = "Good Posture" if pred_value > 0.5 else "Bad Posture"
            confidence = pred_value if pred_value > 0.5 else (1.0 - pred_value)

        result = {
            'prediction': predicted_class,
            'confidence': float(confidence)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/landmarks', methods=['POST'])
def get_landmarks():
    """Extract and return pose landmarks from image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_image)

        if results.pose_landmarks:
            landmarks_data = []
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_data.append({
                    'id': i,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

            return jsonify({
                'landmarks': landmarks_data,
                'count': len(landmarks_data)
            })
        else:
            return jsonify({'error': 'No pose detected'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera/status')
def camera_status():
    """Check camera status"""
    global camera_manager
    return jsonify({
        'is_active': camera_manager.is_active,
        'status': 'active' if camera_manager.is_active else 'inactive'
    })

@app.route('/camera/restart', methods=['POST'])
def restart_camera():
    """Restart camera connection"""
    global camera_manager
    try:
        camera_manager.release_camera()
        success = camera_manager.start_camera()
        return jsonify({
            'success': success,
            'status': 'active' if success else 'failed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cleanup function for when app shuts down
import atexit

def cleanup():
    """Clean up resources on app shutdown"""
    global camera_manager
    camera_manager.release_camera()

atexit.register(cleanup)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
