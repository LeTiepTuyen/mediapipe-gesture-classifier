"""Training script for hand gesture recognition model.

This script trains a 3-layer MLP neural network to classify hand gestures
based on preprocessed hand landmarks extracted from webcam footage.

The training pipeline includes:
- Data loading and preprocessing
- Train/val/test split (60/20/20)
- Model training with early stopping
- Classification metrics and confusion matrix
- Model checkpoint saving
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gesture_model import train_evaluate


def main():
    """Main training function."""
    print("Starting gesture recognition model training...")
    print("=" * 60)
    
    # Train model with updated paths for new src/ structure
    # Data paths are relative to project root:
    # - data/keypoints.csv: Training features
    # - data/labels.csv: Gesture class names
    # Model checkpoint will be saved to: src/models/checkpoints/best_model.pth
    train_evaluate(
        data_path="data/keypoints.csv",
        label_path="data/labels.csv",
        model_save_path="src/models/checkpoints/best_model.pth"
    )


if __name__ == "__main__":
    main()
