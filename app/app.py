import os
import uuid
import threading
from flask import Flask, request, jsonify, render_template, Response, url_for
import logging
import atexit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

from .camera_predictor import CameraPredictor
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configuration ---
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
LEARN_VIDEOS_FOLDER = os.path.join(app.root_path, 'static', 'learn_videos')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LEARN_VIDEOS_FOLDER, exist_ok=True)

# --- Background Task Management ---
tasks = {}

def process_video_task(task_id, video_path):
    try:
        result = predictor.predict_from_video(video_path)
        tasks[task_id] = {'status': 'completed', 'result': result}
    except Exception as e:
        tasks[task_id] = {'status': 'failed', 'error': str(e)}
    finally:
        if os.path.exists(video_path): os.remove(video_path)

# --- Predictor Initialization ---
try:
    predictor = CameraPredictor()
except Exception as e:
    logging.critical(f"CRITICAL ERROR: Failed to initialize CameraPredictor: {e}")
    predictor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Web & API Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/learn-data')
def get_learn_data():
    if predictor is None: return jsonify({"error": "Model not loaded."}), 500
    class_names = predictor.get_class_names()
    learn_data = []
    for name in class_names:
        video_file = next((f for f in os.listdir(LEARN_VIDEOS_FOLDER) if f.lower().startswith(name.lower() + '.')), None)
        learn_data.append({
            "name": name,
            "video_url": url_for('static', filename=f'learn_videos/{video_file}') if video_file else None
        })
    return jsonify(learn_data)

# --- Camera Feed Routes ---
@app.route('/video_feed')
def video_feed():
    if predictor is None: return "Model not loaded", 500
    return Response(predictor.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    """Endpoint to gracefully stop the camera."""
    if predictor:
        predictor.release_camera()
    return jsonify({"status": "Camera feed stopped"})

@app.route('/api/prediction')
def get_prediction():
    if predictor is None: return jsonify({"predicted_sign": "Model Error"}), 500
    return jsonify(predictor.predict_from_buffer())

# --- File Upload Routes ---
@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    if predictor is None: return jsonify({"error": "Model is not loaded."}), 500
    if 'video' not in request.files: return jsonify({"error": "No video file found."}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or no file selected."}), 400
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'processing'}
    thread = threading.Thread(target=process_video_task, args=(task_id, temp_path))
    thread.start()
    return jsonify({"task_id": task_id})

@app.route('/api/result/<task_id>', methods=['GET'])
def get_result(task_id):
    task = tasks.get(task_id)
    if not task: return jsonify({"error": "Invalid Task ID"}), 404
    if task['status'] == 'completed':
        result = task.get('result')
        tasks.pop(task_id, None)
        return jsonify({"status": "completed", "result": result})
    if task['status'] == 'failed':
        error = task.get('error', 'Unknown error')
        tasks.pop(task_id, None)
        return jsonify({"status": "failed", "error": error})
    return jsonify({"status": "processing"})

# --- Cleanup ---
def cleanup():
    if predictor:
        predictor.release_camera()
    for fname in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, fname))
        except: pass
atexit.register(cleanup)

if __name__ == '__main__':
    if predictor is None:
        print("\n" + "="*60 + "\n❌ ERROR: FAILED TO START FLASK SERVER.\n" + "="*60 + "\n")
    else:
        print("\n" + "="*60 + "\n🚀 Starting Drishti AI Flask Server\n   Access in your browser: http://127.0.0.1:5000\n" + "="*60 + "\n")
        app.run(debug=True, host='0.0.0.0')