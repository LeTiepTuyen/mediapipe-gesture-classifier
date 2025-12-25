from pathlib import Path
import cv2 as cv
from mediapipe import solutions
import torch
from utils.preprocessing import calculate_Hands_Coordinates, pre_process_landmark
import sys

# MediaPipe setup
mp_hands = solutions.hands
drawing = solutions.drawing_utils
drawing_styles = solutions.drawing_styles

# Import model class
from Models.gesture_model import Hand_Gesture_Model

# Expose the class at the name pickle expects (for backward compatibility)
sys.modules['__main__'].Hand_Gesture_Model = Hand_Gesture_Model
sys.modules['__main__'].Hand_Gesture = Hand_Gesture_Model


def load_labels(label_path: Path) -> list[str]:
    label_path = Path(label_path)
    try:
        # Use utf-8-sig to automatically strip BOM if present
        labels = [line.strip() for line in label_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    except FileNotFoundError:
        # Fallback to legacy 4 gestures if label file missing
        labels = ["Open", "Close", "Pointer", "OK"]
    return labels


def load_model(checkpoint_path: Path, device: str, labels: list[str]) -> tuple[torch.nn.Module, list[str]]:
    obj = torch.load(checkpoint_path, map_location=device)

    # New checkpoint format with state_dict & metadata
    if isinstance(obj, dict) and "state_dict" in obj and "num_classes" in obj:
        num_classes = obj["num_classes"]
        model = Hand_Gesture_Model(num_classes).to(device)
        model.load_state_dict(obj["state_dict"])
        # Prefer labels stored in checkpoint if available
        chk_labels = obj.get("labels")
        if chk_labels:
            labels = chk_labels
        return model.eval(), labels

    # Legacy format: full model object
    model = obj.to(device)
    if hasattr(model, "f3") and hasattr(model.f3, "out_features"):
        num_classes = model.f3.out_features
        if len(labels) != num_classes:
            labels = labels[:num_classes]
    return model.eval(), labels


def init_webcam(width=640, height=480, max_hands=2, default_webcam=0):
    """Initialize webcam with optimized settings"""
    cap = cv.VideoCapture(default_webcam)
    
    # Set optimized resolution (lower = faster)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    
    # Optimize buffer and FPS
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    cap.set(cv.CAP_PROP_FPS, 30)  # Set target FPS

    # Optimized MediaPipe Hands settings
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0  # Use lighter model (0=lite, 1=full)
    )

    return cap, hands


def functional_hand_hand_gesture_webcam():
    """Real-time hand gesture recognition with optimized performance"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = Path("Models/best_model.pth")
    LABEL_PATH = Path("keypoint_classifier_label.csv")

    labels = load_labels(LABEL_PATH)
    model, labels = load_model(MODEL_PATH, device, labels)

    # Initialize webcam with optimized settings
    cap, hands = init_webcam()
    print("Starting webcam... Press ESC to exit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip for mirror effect
        frame = cv.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process hand detection
        results = hands.process(rgb)

        # Draw landmarks and predict gesture for each detected hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style()
                )

                # Extract and preprocess landmarks
                pts = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                coords = calculate_Hands_Coordinates(frame, pts)
                feat = pre_process_landmark(coords)

                # Predict gesture
                x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    pred_idx = int(torch.argmax(logits, dim=1).item())
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    conf = float(probs[pred_idx])

                # Display prediction on frame
                label = labels[pred_idx] if pred_idx < len(labels) else "Unknown"
                text = f"{label} ({conf:.2f})"
                cv.putText(
                    frame,
                    text,
                    (10, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

        # Display frame outside the for loop
        cv.imshow("Hand Gesture Recognition", frame)

        # Check for ESC key to exit
        if cv.waitKey(1) & 0xFF == 27:
            break

    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    hands.close()
    print("Webcam closed")







