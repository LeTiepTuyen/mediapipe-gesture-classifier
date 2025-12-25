"""Preprocessing utilities for hand landmark data."""
import copy
import numpy as np


def calculate_Hands_Coordinates(image, landmarks):
    """Convert normalized MediaPipe landmarks to pixel coordinates.
    
    MediaPipe returns normalized coordinates (0-1). This function converts them
    to actual pixel positions in the image.
    
    Args:
        image: Input image (H x W x C)
        landmarks: Normalized landmarks from MediaPipe
        
    Returns:
        numpy.ndarray: Landmarks in pixel coordinates
    """
    height, width = image.shape[:2]
    landmarks = np.array(landmarks, dtype=np.float32)
    coordinates = landmarks * np.array([width, height], dtype=np.float32)
    return coordinates


def pre_process_landmark(landmark_list):
    """Normalize landmarks relative to wrist position.
    
    Removes position and scale information by:
    1. Subtracting wrist position (landmark[0])
    2. Normalizing by max absolute value
    
    This makes the features invariant to hand position and size.
    
    Args:
        landmark_list: List of landmark coordinates
        
    Returns:
        numpy.ndarray: Flattened normalized landmarks
    """
    temp_landmark_list = copy.deepcopy(landmark_list)
    temp_landmark_list = np.array(landmark_list, dtype=np.float32)
    
    # Subtract wrist position (first landmark) to normalize
    temp_landmark_list -= temp_landmark_list[0]
    
    # Flatten to 1D array
    temp_landmark_list = temp_landmark_list.flatten()
    
    # Normalize by max absolute value (scale invariant)
    max_value = np.max(np.abs(temp_landmark_list))
    
    if max_value == 0:
        return temp_landmark_list
    
    return temp_landmark_list / max_value
