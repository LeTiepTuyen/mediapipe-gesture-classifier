"""Entry point for hand gesture recognition demo.

This script launches the real-time webcam-based gesture recognition system.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.webcam import functional_hand_hand_gesture_webcam


if __name__ == "__main__":
    functional_hand_hand_gesture_webcam()
