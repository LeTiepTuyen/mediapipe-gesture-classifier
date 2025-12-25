"""Data collection tool for gesture training dataset.

This script captures hand landmarks from webcam and saves them to a CSV file.
The captured data is preprocessed using the same pipeline as inference.

Controls:
  [0..9]   Select label index
  SPACE    Save single sample for current label
  C        Toggle continuous capture (captures ~10 samples/sec)
  R        Reset (delete all samples for current label)
  N / B    Next / Previous label
  H        Show help
  ESC      Exit
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import time
import cv2 as cv
from mediapipe import solutions

from src.utils.preprocessing import calculate_Hands_Coordinates, pre_process_landmark

# Updated paths for new src/ structure
LABEL_FILE = Path("data/labels.csv")
CSV_FILE = Path("data/keypoints.csv")

# Ensure data directory exists
CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load labels dynamically
labels = [line.strip() for line in LABEL_FILE.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
if not labels:
    raise RuntimeError("No labels found in data/labels.csv")

mp_hands = solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=0,
)
drawing = solutions.drawing_utils

HELP = """
Controls:
  [0..9]   Select label index
  SPACE    Save single sample for current label
  C        Toggle continuous capture (captures ~10 samples/sec)
  R        Reset (delete all samples for current label)
  N / B    Next / Previous label
  H        Show help
  ESC      Exit

Labels:
""" + "\n".join([f"  {i}: {name}" for i, name in enumerate(labels)])


def count_samples(label_idx: int) -> int:
    """Count number of samples for a specific label.
    
    Args:
        label_idx: Class index to count samples for
        
    Returns:
        Number of samples found for the label
    """
    if not CSV_FILE.exists():
        return 0
    count = 0
    with CSV_FILE.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and int(row[0]) == label_idx:
                count += 1
    return count


def reset_samples(label_idx: int) -> int:
    """Delete all samples for a specific label.
    
    Args:
        label_idx: Class index to reset
        
    Returns:
        Number of samples deleted
    """
    if not CSV_FILE.exists():
        return 0
    
    rows_to_keep = []
    deleted_count = 0
    
    with CSV_FILE.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and int(row[0]) == label_idx:
                deleted_count += 1
            else:
                rows_to_keep.append(row)
    
    # Rewrite file without deleted samples
    with CSV_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_to_keep)
    
    return deleted_count


def save_row(label_idx: int, landmarks, frame) -> None:
    """Save a single preprocessed gesture sample to CSV.
    
    Args:
        label_idx: Class index for this gesture
        landmarks: MediaPipe hand landmarks
        frame: Current video frame (for coordinate system reference)
    """
    # Extract landmarks as list of [x, y]
    pts = [[lm.x, lm.y] for lm in landmarks]
    
    # Apply SAME preprocessing as webcam.py
    coords = calculate_Hands_Coordinates(frame, pts)
    feat = pre_process_landmark(coords)
    
    # Save preprocessed features
    row = [label_idx] + feat.tolist()
    
    if len(row) != 1 + 42:  # 21 landmarks * 2
        return
    
    with CSV_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    """Main data collection loop."""
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    current_label = 0
    continuous = False
    last_capture = 0.0
    sample_count = count_samples(current_label)  # Track samples for current label
    print(HELP)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand_landmarks = res.multi_hand_landmarks[0]
            drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            now = time.time()
            # Continuous capture ~10 samples/sec
            if continuous and (now - last_capture) > 0.1:
                save_row(current_label, hand_landmarks.landmark, frame)
                sample_count += 1
                last_capture = now

        # Overlay UI with sample count
        cv.putText(frame, f"Label: {current_label} - {labels[current_label]}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, f"Samples: {sample_count}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, f"Continuous: {'ON' if continuous else 'OFF'}", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, "SPACE=save | C=toggle | R=reset | N/B=label | ESC=quit", (10, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)

        cv.imshow("Collect Gestures", frame)
        key = cv.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('h') or key == ord('H'):
            print(HELP)
        elif key == ord('c') or key == ord('C'):
            continuous = not continuous
            last_capture = 0.0
        elif key == ord('r') or key == ord('R'):
            # Reset samples for current label
            deleted = reset_samples(current_label)
            sample_count = 0
            print(f"Reset: Deleted {deleted} samples for label {current_label} - {labels[current_label]}")
        elif key == ord('n') or key == ord('N'):
            current_label = (current_label + 1) % len(labels)
            sample_count = count_samples(current_label)  # Update count for new label
        elif key == ord('b') or key == ord('B'):
            current_label = (current_label - 1) % len(labels)
            sample_count = count_samples(current_label)  # Update count for new label
        elif key == ord(' '):  # SPACE save single sample
            if res.multi_hand_landmarks:
                save_row(current_label, res.multi_hand_landmarks[0].landmark, frame)
                sample_count += 1
        elif ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx < len(labels):
                current_label = idx
                sample_count = count_samples(current_label)  # Update count for new label

    cap.release()
    cv.destroyAllWindows()
    hands.close()
    print("Done. Samples appended to", CSV_FILE)


if __name__ == "__main__":
    main()
