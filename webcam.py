from pathlib import Path
import cv2 as cv
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import torch
from utils.preprocessing import calculate_Hands_Coordinates, pre_process_landmark
import sys

# MediaPipe setup
mp_hands = solutions.hands
drawing = solutions.drawing_utils
drawing_styles = solutions.drawing_styles

# Import model class
from Models.gesture_model import Hand_Gesture_Model

# Expose the class at the name pickle expects
sys.modules['__main__'].Hand_Gesture_Model = Hand_Gesture_Model
sys.modules['__main__'].Hand_Gesture = Hand_Gesture_Model


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
    
    # Load model once
    MODEL_PATH = Path("Models/best_model.pth")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    
    # Initialize webcam with optimized settings
    cap, hands = init_webcam()
    label_map = {0: "Open", 1: "Close", 2: "Pointer", 3: "OK"}
    
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
                text = f"{label_map.get(pred_idx, 'Unknown')} ({conf:.2f})"
                cv.putText(frame, text, (10, 40),
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        
        # CRITICAL FIX: Display frame outside the for loop
        cv.imshow("Hand Gesture Recognition", frame)
        
        # Check for ESC key to exit
        if cv.waitKey(1) & 0xFF == 27:
            break
    
    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    hands.close()
    print("Webcam closed")







