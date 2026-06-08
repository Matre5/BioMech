import numpy as np

def calculate_angle(A, B, C):
    """
    Calculate the angle at point B (the vertex)
    A, B, C are each tuples of (x, y) coordinates
    """
    
    vector_1 = np.array(A) - np.array(B)
    vector_2 = np.array(C) - np.array(B)

    V1_V2 = np.dot(vector_1, vector_2)
    
    V1_mag = np.linalg.norm(vector_1)
    V2_mag = np.linalg.norm(vector_2)
    
    cos_angle = V1_V2/ (V1_mag * V2_mag)
    
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)

    angle_deg = np.degrees(angle)
    
    return angle_deg

