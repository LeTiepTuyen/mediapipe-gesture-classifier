import copy
import numpy as np
def calculate_Hands_Coordinates (image,landmarks) :
    '''Finding the actual pixel postions beacuse mediapipe gives the normilzed postion'''
    height, width = image.shape[:2]
    landmarks = np.array(landmarks, dtype=np.float32)
    Coordinates = landmarks * np.array([width, height], dtype=np.float32)
    return Coordinates
def pre_process_landmark(landmark_list):
    '''removing postion and hand size by isbtartiona nd normialzion '''
    temp_landmark_list = copy.deepcopy(landmark_list)
    temp_landmark_list= np.array(landmark_list, dtype=np.float32)
    temp_landmark_list-= temp_landmark_list[0]
    temp_landmark_list = temp_landmark_list.flatten()
    max_value=np.max(np.abs(temp_landmark_list))  # here we use abs becuase if we have values at teh lef and right the using the disturtt value would disturt the distance
    return temp_landmark_list/max_value


