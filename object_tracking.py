import numpy as np
from filterpy.kalman import KalmanFilter

def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.F = np.array([[1, 0, 0, 1, 0, 0], 
                     [0, 1, 0, 0, 1, 0], 
                     [0, 0, 1, 0, 0, 1], 
                     [0, 0, 0, 1, 0, 0], 
                     [0, 0, 0, 0, 1, 0], 
                     [0, 0, 0, 0, 0, 1]])

    kf.H = np.array([[1, 0, 0, 0, 0, 0], 
                     [0, 1, 0, 0, 0, 0], 
                     [0, 0, 1, 0, 0, 0]])

    kf.P *= 500  # Reduce initial uncertainty to avoid erratic predictions
    kf.R = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])  # Reduce measurement noise
    kf.Q = np.eye(6) * 0.05  # Lower process noise for smoother tracking
    
    return kf

def predict_target_position(kf, measurement):
    kf.predict()
    kf.update(measurement)
    return kf.x[:3].flatten()
