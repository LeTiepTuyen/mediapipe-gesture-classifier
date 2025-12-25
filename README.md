# Real-time Hand Gesture Detection - AI Final Project

**Team: Chill Out**

Real-time hand gesture recognition system using MediaPipe and PyTorch. Detects 9 different hand gestures with 97% accuracy using a lightweight 3-layer neural network.

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.9.1-orange.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.13-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Final Project - Artificial Intelligence Course**  
VN-UK Institute For Research and Executive Education (VNUK) - The University of Danang
Academic Year: 2025-2026

## 📹 Demo Video

> **[Watch Full Demo Video on Google Drive](https://drive.google.com/drive/folders/1CusmPW39qA6HPP1uqTYT69TFpAeTXYLe?usp=sharing)**  


## 📸 Demo Screenshots

<p align="center">
  <img src="demo\collect_samples_demo.png" alt="Data Collection Interface" width="45%"/>
  <img src="demo\gesture_detection_demo.png" alt="Real-time Detection" width="45%"/>
  <img src="demo\gesture_detection_demo2.png" alt="Real-time Detection2" width="45%"/>
</p>

*Screenshots showing: (Left) Real-time data collection interface with sample counter, (Right) Live gesture detection with confidence scores*



## 👥 Team Members

| Name | Student ID | Role |
|------|------------|------|
| **Le Tiep Tuyen** | 22020015 | Team Lead & Developer |
| **Mai Thieu Tin** | 22020003 | Developer |
| **Doan Hong Ngoc** | 22020010 | Developer |

## ✨ Features

- **Real-time Recognition**: Optimized for 640x480@30fps webcam performance
- **High Accuracy**: 97% accuracy on test set with 9 gesture classes
- **Lightweight Model**: Simple 3-layer neural network (42 → 22 → 11 → 9)
- **MediaPipe Integration**: Fast and robust hand tracking with 21 landmarks
- **Professional Architecture**: Organized src/ structure with clear separation of concerns
- **Easy Extension**: Simple workflow to add new gestures with custom sample collection
- **Data Preprocessing**: Consistent pipeline between training and inference
- **Model Checkpointing**: Automatic saving of best model during training

## 🎯 Supported Gestures (9 Classes)

| Gesture | Index | Description |
|---------|-------|-------------|
| Open | 0 | Open hand with all fingers extended |
| Close | 1 | Closed fist |
| Pointer | 2 | Index finger pointing |
| OK | 3 | Thumb and index finger touching |
| Thumbs Up | 4 | Thumb extended upward |
| Thumbs Down | 5 | Thumb extended downward |
| Peace | 6 | Index and middle fingers extended (V-sign) |
| Rock | 7 | Index and pinky fingers extended |
| Stop | 8 | Open hand facing forward |

## 📁 Project Structure

```
mediapipe-gesture-classifier/
├── src/                          # Source code
│   ├── app/                      # Application entry points
│   │   ├── main.py              # Demo launcher
│   │   └── webcam.py            # Real-time recognition
│   ├── models/                   # Neural network models
│   │   ├── gesture_model.py     # Model architecture & training
│   │   └── checkpoints/         # Saved model weights
│   │       └── best_model.pth   # Best trained model
│   └── utils/                    # Utility functions
│       └── preprocessing.py     # Landmark preprocessing
├── scripts/                      # Standalone scripts
│   ├── collect_data.py          # Data collection tool
│   └── train_model.py           # Training script
├── data/                         # Dataset
│   ├── keypoints.csv            # Training features (3856 samples)
│   ├── labels.csv               # Gesture class names
│   └── backups/                 # Data backups
├── demo/                         # Demo screenshots and videos
├── requirements.txt              # Python dependencies
├── Pipfile                       # Pipenv configuration
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

#### Prerequisites
- Python 3.12 or higher
- Webcam for real-time detection
- (Optional) CUDA-compatible GPU for faster training

#### Clone Repository
```bash
git clone https://github.com/your-username/mediapipe-gesture-classifier.git
cd mediapipe-gesture-classifier
```

#### Install Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Or using pipenv
pipenv install
pipenv shell
```

### 2. Run Real-time Detection

Run the pre-trained model for live gesture recognition:

```bash
python src/app/main.py
```

**Controls:**
- Show your hand to the webcam
- Gesture prediction and confidence score will appear on screen
- Press **ESC** to exit

### 3. Model Training (Optional)

If you want to retrain the model with the existing dataset:

```bash
python scripts/train_model.py
```

**Training Details:**
- Dataset: 3856 samples (336-465 per gesture)
- Split: 60% train / 20% validation / 20% test
- Architecture: 3-layer MLP (42 → 22 → 11 → 9)
- Optimizer: Adam (lr=0.001)
- Batch Size: 128
- Early Stopping: Patience of 5 epochs
- Expected Accuracy: ~97%

## 📝 Adding New Gestures

Want to add your own custom gestures? Follow this complete workflow:

### Step 1: Add Gesture Label

Add your new gesture name to the labels file:

```bash
# Edit data/labels.csv
# Add your gesture name on a new line, for example:
Open
Close
Pointer
OK
Thumbs_Up
Thumbs_Down
Peace
Rock
Stop
Victory        # <-- Your new gesture
```

**Important:** 
- Gesture names should be descriptive
- Use underscores for multi-word names (e.g., `Thumbs_Up`)
- Each gesture should be on a separate line
- The order determines the class index (0, 1, 2, ...)

### Step 2: Collect Sample Data

Launch the data collection tool:

```bash
python scripts/collect_data.py
```

**Data Collection Interface:**

The GUI will display:
- Current gesture label and index
- Sample count for current gesture
- Continuous capture mode status
- Hand landmarks overlay

**Controls:**
- **[0-9]**: Select gesture by index number
- **SPACE**: Capture single sample
- **C**: Toggle continuous capture mode (~10 samples/sec)
- **R**: Reset (delete all samples for current gesture)
- **N / B**: Navigate to next/previous gesture
- **H**: Show help menu
- **ESC**: Exit and save

**Collection Workflow:**
1. Press the number key for your new gesture (e.g., `9` for Victory)
2. Position your hand clearly in the webcam frame
3. Press **C** to start continuous capture
4. Perform the gesture from different angles and positions
5. Collect **300-500 samples** for best results
6. Press **C** again to stop capture
7. Review the sample count displayed on screen

> **💡 Tip: Sample Collection Best Practices**
> - Collect samples with varied hand positions (center, left, right, top, bottom)
> - Include different angles and rotations
> - Vary the distance from camera (near and far)
> - Use different lighting conditions if possible
> - **Recommended: 300-500 samples per gesture for optimal accuracy**
> - You can adjust sample counts based on gesture complexity:
>   - Simple gestures (Open, Close): 200-300 samples
>   - Complex gestures (OK, Peace): 400-500 samples
> - **Note**: You can customize the number of training samples for any specific gesture based on your needs. More samples generally lead to better accuracy, especially for gestures that are visually similar to others.

### Step 3: Retrain Model

After collecting data for your new gesture, retrain the model:

```bash
python scripts/train_model.py
```

The training script will:
- Automatically detect the new gesture from `data/labels.csv`
- Adjust model output layer to accommodate new class
- Train with stratified train/val/test split
- Save best model to `src/models/checkpoints/best_model.pth`
- Display accuracy metrics and confusion matrix

### Step 4: Test New Gesture

Run the detection demo to test your new gesture:

```bash
python src/app/main.py
```

Perform your new gesture in front of the camera and verify detection accuracy.

### Step 5: Iterate (If Needed)

If accuracy is low:
1. Collect more diverse samples (especially edge cases)
2. Check for similar-looking gestures causing confusion
3. Review confusion matrix from training output
4. Consider adjusting sample distribution per class

## 🔧 Advanced Usage

### Data Preprocessing Pipeline

The system applies consistent preprocessing to ensure training and inference data match:

```python
# 1. Convert normalized landmarks to pixel coordinates
coords = calculate_Hands_Coordinates(frame, landmarks)

# 2. Normalize by wrist position (first landmark)
coords = coords - coords[0]

# 3. Scale by maximum absolute value
max_val = max(abs(coords))
coords = coords / max_val if max_val > 0 else coords

# 4. Flatten to 42-dimensional feature vector
features = coords.flatten()  # Shape: (42,)
```

### Model Architecture

```python
Hand_Gesture_Model(
  (f1): Linear(in_features=42, out_features=22)
  (f2): Linear(in_features=22, out_features=11)
  (f3): Linear(in_features=11, out_features=9)  # Output: num_classes
  (dropout): Dropout(p=0.2)
  (relu): ReLU()
)
```

### Custom Training Parameters

Modify training hyperparameters in `scripts/train_model.py`:

```python
train_evaluate(
    data_path="data/keypoints.csv",
    label_path="data/labels.csv",
    model_save_path="src/models/checkpoints/best_model.pth",
    epochs=200,          # Maximum epochs
    patience=5,          # Early stopping patience
    batch_size=128,      # Batch size
    lr=0.001,           # Learning rate
)
```

### Using Different Model Checkpoint

Specify custom model path in `src/app/webcam.py`:

```python
MODEL_PATH = Path("src/models/checkpoints/your_model.pth")
```

## 📊 Performance Metrics

### Training Results (9 Gestures, 3856 Samples)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.0% |
| **Training Time** | ~2-3 minutes (CPU) |
| **Inference Speed** | 30 FPS @ 640x480 |
| **Model Size** | 8.2 KB |

### Per-Class Performance

| Gesture | Precision | Recall | F1-Score | Samples |
|---------|-----------|--------|----------|---------|
| Open | 0.94 | 0.84 | 0.89 | 336 |
| Close | 1.00 | 0.99 | 0.99 | 363 |
| Pointer | 1.00 | 1.00 | 1.00 | 419 |
| OK | 1.00 | 1.00 | 1.00 | 422 |
| Thumbs_Up | 1.00 | 1.00 | 1.00 | 432 |
| Thumbs_Down | 0.95 | 1.00 | 0.98 | 442 |
| Peace | 0.99 | 1.00 | 0.99 | 450 |
| Rock | 1.00 | 0.99 | 0.99 | 459 |
| Stop | 0.85 | 0.92 | 0.88 | 465 |

## 🛠️ Technology Stack

- **Python 3.12.7**: Core programming language
- **PyTorch 2.9.1**: Neural network framework
- **MediaPipe 0.10.13**: Hand tracking and landmark detection
- **OpenCV 4.12.0**: Computer vision and webcam capture
- **NumPy**: Numerical operations
- **Pandas**: Data handling
- **Scikit-learn**: Train/test split and metrics

## 📚 Technical Details

### Hand Landmark Detection

MediaPipe detects 21 landmarks per hand:
- Wrist (1 point)
- Thumb (4 points)
- Index finger (4 points)
- Middle finger (4 points)
- Ring finger (4 points)
- Pinky finger (4 points)

Each landmark provides (x, y) coordinates → 42 features total.

### Data Format

**`data/keypoints.csv`**:
```csv
label_index,x0,y0,x1,y1,x2,y2,...,x20,y20
0,0.0,0.0,-0.206,-0.071,...,-0.800
1,0.0,0.0,-0.210,-0.073,...,-0.787
```

**`data/labels.csv`**:
```csv
Open
Close
Pointer
...
```

### Training Pipeline

1. **Data Loading**: Read CSV files with UTF-8 BOM handling
2. **Preprocessing**: Apply landmark normalization
3. **Split**: Stratified 60/20/20 train/val/test split
4. **Training**: Adam optimizer with early stopping
5. **Evaluation**: Classification report and confusion matrix
6. **Checkpointing**: Save best model based on validation loss

## 🐛 Troubleshooting

### Issue: Webcam not detected
**Solution**: Check camera permissions and ensure no other application is using the webcam.

### Issue: Import errors
**Solution**: Ensure you're running from project root directory:
```bash
cd mediapipe-gesture-classifier
python src/app/main.py
```

### Issue: Low detection accuracy
**Solution**: 
- Ensure good lighting conditions
- Position hand clearly in frame
- Collect more training samples for problematic gestures
- Check if gestures are too similar (review confusion matrix)

### Issue: Slow performance
**Solution**:
- Reduce webcam resolution in `src/app/webcam.py`
- Decrease MediaPipe `model_complexity` (already set to 0)
- Close other CPU-intensive applications

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking technology
- **PyTorch** team for deep learning framework
- **USTH Faculty** for guidance and support
- **OpenCV** community for computer vision tools

## 📧 Contact

For questions or collaboration:
- **Team Lead**: Le Tiep Tuyen
- **Email**: [tuyentieple@gmail.com](mailto:tuyentieple@gmail.com)
- **GitHub**: [https://github.com/LeTiepTuyen/mediapipe-gesture-classifier](https://github.com/LeTiepTuyen/mediapipe-gesture-classifier?tab=readme-ov-file)

---

**Built with ❤️ by Team Chill Out**  
*VN-UK Institute For Research and Executive Education (VNUK) - The University of Danang*  
*Academic Year 2025-2026*
