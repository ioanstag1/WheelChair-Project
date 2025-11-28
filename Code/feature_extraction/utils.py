import numpy as np
from scipy.signal import savgol_filter

# ======================
# GEOMETRY FUNCTIONS
# ======================

def calculate_trunk_pitch_angle(hip, neck, y_down=True, right=False):
    dx = neck[0] - hip[0]
    dy = neck[1] - hip[1]
    angle_rad = np.arctan2(dx, -dy) if y_down else np.arctan2(dx, dy)
    angle_deg = np.degrees(angle_rad)
    return angle_deg if right else -angle_deg

def calculate_knee_angle(hip, knee, side):
    vector_hip_to_knee = knee - hip
    if side == "left":
        vector_hip_to_knee = -vector_hip_to_knee
    angle_rad = np.arctan2(vector_hip_to_knee[1], vector_hip_to_knee[0])
    return np.degrees(angle_rad)

def process_knee_angles(side, hip, knee):
    if side == "right":
        return calculate_knee_angle(hip, knee, side)
    elif side == "left":
        return -calculate_knee_angle(hip, knee, side)
    return None

def calculate_elbow_angle(shoulder, elbow, wrist):
    v1 = elbow - shoulder
    v2 = wrist - elbow
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0: return 0.0
    cos_theta = np.dot(v1, v2) / denom
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def calculate_shoulder_asymmetry(left_shoulder, right_shoulder):
    L = np.asarray(left_shoulder, float)
    R = np.asarray(right_shoulder, float)
    if L[0] > R[0]: L, R = R, L
    dx = R[0] - L[0]
    dy = R[1] - L[1]
    ang = np.degrees(np.arctan2(dy, dx))
    ang = ((ang + 90) % 180) - 90
    return -ang 

def calculate_hand_height_asymmetry(left_wrist, right_wrist, left_shoulder, right_shoulder):
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist == 0: return 0.0
    dy = abs(left_wrist[1] - right_wrist[1])
    return dy / shoulder_dist

def calculate_wrist_normalized_positions(left_shoulder, right_shoulder, left_wrist, right_wrist):
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    if shoulder_width == 0: return 0.0, 0.0
    left_wrist_norm = (left_wrist[1] - left_shoulder[1]) / shoulder_width
    right_wrist_norm = (right_wrist[1] - right_shoulder[1]) / shoulder_width
    return left_wrist_norm, right_wrist_norm

def calculate_trunk_roll(neck, hip):
    dx = neck[0] - hip[0]
    dy = neck[1] - hip[1]
    roll = np.degrees(np.arctan2(dy, dx)) - 90
    return ((roll + 90) % 180) - 90

def calculate_head_roll(left_ear, right_ear):
    L, R = np.asarray(left_ear, float), np.asarray(right_ear, float)
    dx = R[0] - L[0]
    dy = R[1] - L[1]
    ang = np.degrees(np.arctan2(dy, dx))
    ang = ((ang + 90) % 180) - 90
    return -ang

def calculate_head_pitch(neck, head, y_down=True, right=False):
    dx = head[0] - neck[0]
    dy = head[1] - neck[1]
    ang = np.degrees(np.arctan2(dx, -dy) if y_down else np.arctan2(dx, dy))
    return ang if right else -ang

def calculate_pelvis_pitch(spine_03, hip, y_down=True, right=False):
    dx = spine_03[0] - hip[0]
    dy = spine_03[1] - hip[1]
    ang = np.degrees(np.arctan2(dx, -dy) if y_down else np.arctan2(dx, dy))
    return ang if right else -ang

# ======================
# SIGNAL PROCESSING
# ======================

def smooth(y):
    win = 21 if len(y) > 21 else len(y) - (len(y) % 2 == 0)
    if win < 5: return y
    return savgol_filter(y, win, 3)

def derivative(y, dt):
    return np.gradient(y, dt)

def compute_sparc(signal, fs):
    vel = derivative(signal, 1/fs)
    mag = np.abs(np.fft.rfft(vel))
    if np.sum(mag) == 0: return 0.0
    mag /= np.sum(mag)
    log_mag = np.log(mag + 1e-8)
    freq = np.linspace(0, fs/2, len(log_mag))
    return -np.trapezoid(mag * log_mag, freq)