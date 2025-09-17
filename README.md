# Indian Sign Language Recognition using XGBoost

## Problem Statement

Indian Sign Language (ISL) recognition is crucial for bridging communication gaps between deaf/hard-of-hearing individuals and the general population. This project aims to develop an automated system that can recognize ISL signs from video sequences using machine learning techniques.

The main challenges include:
- Variable-length sign sequences with different temporal dynamics
- Limited labeled data for training robust models
- Real-time processing requirements for practical applications
- Handling diverse signers with different signing styles and speeds

## Approach

Our solution uses a **feature engineering + XGBoost** approach that achieves ~68% accuracy:

### 1. **Landmark Extraction**
- Use MediaPipe Holistic to extract 1040-dimensional landmark features per frame:
  - Right hand: 21 points × 2 coordinates = 42 features
  - Left hand: 21 points × 2 coordinates = 42 features  
  - Face: 468 points × 2 coordinates = 936 features (with refined landmarks)

### 2. **Feature Engineering**
- Convert variable-length sequences to fixed-size feature vectors using statistical summaries:
  - **Mean**: Average landmark positions across time
  - **Standard Deviation**: Variability of movements
  - **Minimum**: Extreme positions in one direction
  - **Maximum**: Extreme positions in the other direction
- Final feature vector: 1040 × 4 = 4160 features per sign sequence

### 3. **Classification**
- Use XGBoost (Extreme Gradient Boosting) for multi-class classification
- Handles class imbalance and non-linear feature interactions effectively
- Fast training and inference suitable for real-time applications

### 4. **Advantages of This Approach**
- **Robust**: Statistical features are less sensitive to temporal variations
- **Fast**: Training completes in minutes, inference in milliseconds
- **Interpretable**: Tree-based model provides feature importance insights
- **Scalable**: Easy to add new sign classes

## Dataset

We use the **INCLUDE dataset** - a large-scale Indian Sign Language dataset:

- **Size**: 447 video samples across 29 sign classes
- **Categories**: Animals, Clothes, House objects, etc.
- **Format**: MP4 videos with corresponding landmark files
- **Signs**: Horse, Mouse, Dress, Hat, Shirt, Soap, Table, etc.

### Download INCLUDE Dataset

**Official Links:**
- **Hugging Face**: https://huggingface.co/datasets/ai4bharat/INCLUDE
- **Zenodo**: https://zenodo.org/records/4010759
- **GitHub Repository**: https://github.com/AI4Bharat/INCLUDE

**Download Script:**
```bash
#!/bin/bash
base_url="https://zenodo.org/api/records/4010759"
response=$(curl -s "$base_url")
echo "$response" | jq -r '.files[] | .links.self + " " + .key' | while read -r file_url file_name
do
    echo "Downloading $file_name from $file_url..."
    curl -o "$file_name" "$file_url"
    echo "$file_name downloaded."
done
for file in *.zip; do
    unzip "${file%.zip}"
done
```

## Folder Structure

### Expected Directory Layout

```
Data/INCLUDE/
├── Adjectives/
│   └── Landmarks/              # ⚠️ CREATE THIS MANUALLY
│       └── [Sign_Name]/        # e.g., Good, Bad
│           ├── video1.npy
│           ├── video2.npy
│           └── ...
├── Animals/
│   └── Landmarks/              # ⚠️ CREATE THIS MANUALLY  
│       ├── Horse/
│       │   ├── horse_001.npy
│       │   ├── horse_002.npy
│       │   └── ...
│       └── Mouse/
│           ├── mouse_001.npy
│           ├── mouse_002.npy
│           └── ...
├── Clothes/
│   └── Landmarks/              # ⚠️ CREATE THIS MANUALLY
│       ├── Dress/
│       ├── Hat/
│       ├── Shirt/
│       └── ...
├── House/
│   └── Landmarks/              # ⚠️ CREATE THIS MANUALLY
│       ├── Bathroom/
│       ├── Bedroom/  
│       ├── Kitchen/
│       └── ...
└── [Other_Categories]/
    └── Landmarks/              # ⚠️ CREATE THIS MANUALLY
        └── [Sign_Names]/
            └── *.npy files
```

### ⚠️ IMPORTANT: Manual Setup Required

**The INCLUDE dataset only provides Videos folders. You must:**

1. **Create `Landmarks` folders** inside each category folder
2. **Create subfolders** for each sign name inside `Landmarks`
3. **Run landmark extraction** to generate `.npy` files from videos

### Example Commands:
```bash
# Navigate to your INCLUDE directory
cd Data/INCLUDE

# Create Landmarks folders for each category
for category in */; do
    mkdir -p "${category}Landmarks"
    echo "Created ${category}Landmarks/"
done

# For each sign, create subfolder and extract landmarks
# (You'll need to run MediaPipe extraction script)
```

## Implementation

### Requirements
```bash
pip install xgboost scikit-learn numpy matplotlib mediapipe opencv-python
```

### Training the Model
```python
python model_xgb.py
```

### Key Files
- `model_xgb.py` - Main XGBoost training and evaluation script
- `landmark_extraction.py` - MediaPipe landmark extraction from videos
- `README.md` - This documentation

### Expected Output
```
Found 447 samples across 29 sign classes
Accuracy: 0.6777

Sample predictions:
True: Horse           Predicted: Horse
True: Mouse           Predicted: Mouse  
True: Dress           Predicted: Hat
...
```

## Results

- **Accuracy**: ~68% on 29-class ISL recognition
- **Training Time**: ~2-3 minutes
- **Inference Time**: <10ms per sign
- **Model Size**: <50MB

### Performance by Category
The model performs well on:
- **Animals** (Horse, Mouse): Clear hand movements
- **Objects** (Table, Chair): Distinctive gestures
- **Clothing** (Dress, Hat): Simple motions

Challenging signs:
- Similar hand shapes (Pen vs Pencil)
- Complex sequences (Dream, Paint)

## Future Improvements

1. **Enhanced Feature Engineering**
   - Velocity/acceleration features (frame-to-frame differences)
   - Hand distance metrics (distance between left/right hands)
   - Joint angle computations

2. **Data Augmentation**
   - Temporal jittering (varying playback speed)
   - Spatial noise injection
   - Mirror flipping for symmetric signs

3. **Real-Time Deployment**
   - Integrate with live camera feed
   - Optimize inference pipeline
   - Add confidence thresholding

4. **Deep Learning Comparison**
   - Compare against LSTM/GRU models
   - Explore Transformer architectures
   - Investigate 3D CNNs

## References

- **INCLUDE Dataset Paper**: [A Large Scale Dataset for Indian Sign Language Recognition](https://dl.acm.org/doi/10.1145/3394171.3413528)
- **MediaPipe Holistic**: https://developers.google.com/mediapipe/solutions/vision/holistic
- **XGBoost Documentation**: https://xgboost.readthedocs.io/

## Citation

If you use this work, please cite:
```
@article{include_dataset_2020,
  title={INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
  author={Sridhar, Advaith and Ganesan, Rohith Gandhi and Kumar, Pratyush and Khapra, Mitesh},
  journal={Proceedings of the 28th ACM International Conference on Multimedia},
  year={2020}
}
```