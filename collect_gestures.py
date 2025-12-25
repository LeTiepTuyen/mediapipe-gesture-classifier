from pathlib import Path
import csv
import time
import cv2 as cv
from mediapipe import solutions

LABEL_FILE = Path("keypoint_classifier_label.csv")
CSV_FILE = Path("keypoint.csv")

# Load labels dynamically
labels = [line.strip() for line in LABEL_FILE.read_text().splitlines() if line.strip()]
if not labels:
    raise RuntimeError("No labels found in keypoint_classifier_label.csv")

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
  N / B    Next / Previous label
  H        Show help
  ESC      Exit

Labels:
""" + "\n".join([f"  {i}: {name}" for i, name in enumerate(labels)])


def save_row(label_idx: int, landmarks) -> None:
    row = [label_idx]
    for lm in landmarks:
        row.extend([lm.x, lm.y])  # normalized coords
    if len(row) != 1 + 42:  # 21 landmarks * 2
        return
    with CSV_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    current_label = 0
    continuous = False
    last_capture = 0.0
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
                save_row(current_label, hand_landmarks.landmark)
                last_capture = now

        # Overlay UI
        cv.putText(frame, f"Label: {current_label} - {labels[current_label]}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, f"Continuous: {'ON' if continuous else 'OFF'}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, "SPACE=save, C=toggle, N/B=label, ESC=quit", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        cv.imshow("Collect Gestures", frame)
        key = cv.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('h') or key == ord('H'):
            print(HELP)
        elif key == ord('c') or key == ord('C'):
            continuous = not continuous
            last_capture = 0.0
        elif key == ord('n') or key == ord('N'):
            current_label = (current_label + 1) % len(labels)
        elif key == ord('b') or key == ord('B'):
            current_label = (current_label - 1) % len(labels)
        elif key == ord(' '):  # SPACE save single sample
            if res.multi_hand_landmarks:
                save_row(current_label, res.multi_hand_landmarks[0].landmark)
        elif ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx < len(labels):
                current_label = idx

    cap.release()
    cv.destroyAllWindows()
    hands.close()
    print("Done. Samples appended to", CSV_FILE)


if __name__ == "__main__":
    main()
