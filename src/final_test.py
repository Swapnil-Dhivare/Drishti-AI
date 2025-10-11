import argparse
from pathlib import Path
import sys

# Add the app directory to the path so we can import CameraPredictor
app_dir = Path(__file__).parent.parent / 'app'
sys.path.append(str(app_dir.resolve()))
try:
    from camera_predictor import CameraPredictor
except ModuleNotFoundError:
    print(f"❌ FATAL ERROR: Could not find 'camera_predictor.py' in '{app_dir.resolve()}'. Please ensure the file exists.")
    sys.exit(1)

def main():
    """
    This script provides a definitive test of the end-to-end video prediction pipeline.
    It takes a video file, extracts landmarks, and predicts using the exact
    same code as the Flask web application.
    """
    parser = argparse.ArgumentParser(description="Run a full prediction on a single video file.")
    parser.add_argument("video_file", type=str, help="Path to the video file you want to test.")
    args = parser.parse_args()
    
    video_path = Path(args.video_file)

    print("="*60)
    print("        🚀 Final Model and Prediction Pipeline Verification 🚀        ")
    print("="*60)
    
    if not video_path.exists():
        print(f"❌ FATAL ERROR: Video file not found at '{video_path.resolve()}'")
        return

    try:
        print("✅ Initializing CameraPredictor (loading model)...")
        predictor = CameraPredictor()
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not initialize CameraPredictor. Error: {e}")
        return

    print(f"✅ Found test video: {video_path.name}")
    print("\n--- Running End-to-End Prediction ---")
    
    try:
        # Use the all-in-one prediction method
        result = predictor.predict_from_video(str(video_path))

        print("\n[FINAL RESULT]")
        print(f"  - ✅ Predicted Sign: '{result['predicted_sign']}'")
        print(f"  - Confidence: {result['confidence']:.4f}")
        
        print("\n  - Top 5 Probabilities:")
        for alt in result.get("alternatives", [])[:5]:
            print(f"    - {alt['class']}: {alt['probability']:.4f}")
        print("="*60)
        
        # Check for a common test case
        if 'daughter' in video_path.name.lower() and result['predicted_sign'].lower() == 'daughter':
            print("\n🎉 SUCCESS! The model correctly predicted 'Daughter'. Your pipeline is working!")
        elif 'daughter' in video_path.name.lower():
             print("\n⚠️ ATTENTION: The prediction is still incorrect. This points to a fundamental issue in the saved model pipeline. Please re-run the `src/model.py` script one more time to be certain.")


    except Exception as e:
        print(f"❌ An error occurred during video processing or prediction: {e}")

if __name__ == "__main__":
    main()