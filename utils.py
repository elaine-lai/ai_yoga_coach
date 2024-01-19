import numpy as np
import cv2

keypoints = [[0.3923125, 0.5542421, 0.5562568],
             [0.3295156, 0.60111874, 0.58333707],
             [0.34902796, 0.5428777, 0.54064333],
             [0.3278386, 0.74546766, 0.5753689],
             [0.34782788, 0.60988575, 0.5702256],
             [0.5114173, 0.95146173, 0.6981182],
             [0.5708685, 0.59803444, 0.7739016],
             [0.8679228, 0.8845679, 0.27316803],
             [0.83263266, 0.46695566, 0.5313083],
             [0.6359774, 0.65353507, 0.01975938],
             [0.80994785, 0.3590023, 0.12591188],
             [0.9310886, 0.9022478, 0.0241341],
             [0.9304087, 0.56741756, 0.02165917],
             [0.82178545, 0.7025744, 0.02470911],
             [0.8333831, 0.4202083, 0.05160204],
             [0.85987794, 0.28691214, 0.01190803],
             [0.8683524, 0.24162577, 0.02748439]]

# [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]).

ANGLE_DICT = {
    'left_elbow': ['left_shoulder', 'left_elbow', 'left_wrist'],
    'left_armpit': ['left_elbow', 'left_shoulder', 'left_hip'],
    'left_waist': ['left_shoulder', 'left_hip', 'left_knee'],
    'left_knee': ['left_hip', 'left_knee', 'left_ankle'],
    'right_elbow': ['right_shoulder', 'right_elbow', 'right_wrist'],
    'right_armpit': ['right_elbow', 'right_shoulder', 'right_hip'],
    'right_waist': ['right_shoulder', 'right_hip', 'right_knee'],
    'right_knee': ['right_hip', 'right_knee', 'right_ankle']
}


def label_keypoints(keypoints):
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    return left_elbow, right_elbow, left_shoulder, right_shoulder, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle


def calculate_angle(point1, point2, point3):
    vector1 = np.array([point1[1] - point2[1], point1[0] - point2[0]])
    vector2 = np.array([point3[1] - point2[1], point3[0] - point2[0]])

    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    angle_rad = np.arccos(dot_product / norm_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def get_all_angles(keypoints):
    left_elbow, right_elbow, left_shoulder, right_shoulder, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle = label_keypoints(
        keypoints)
    angle_left_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angle_right_elbow = calculate_angle(right_shoulder, right_elbow, right_wrist)
    angle_left_armpit = calculate_angle(left_elbow, left_shoulder, left_hip)
    angle_right_armpit = calculate_angle(right_elbow, right_shoulder, right_hip)
    angle_left_waist = calculate_angle(left_shoulder, left_hip, left_knee)
    angle_right_waist = calculate_angle(right_shoulder, right_hip, right_knee)
    angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
    angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
    return angle_left_elbow, angle_right_elbow, angle_left_armpit, angle_right_armpit, angle_left_waist, angle_right_waist, angle_left_knee, angle_right_knee


def draw_keypoints(frame, keypoints, confidence_threshold):
    """
    draw keypoints
    :param frame:
    :param keypoints:
    :param confidence_threshold:
    :return:
    """
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        # if the confidence is above confidence threshold, draw keypoint
        if kp_conf > confidence_threshold:
            size_of_circle = 6
            color_of_circle = (0, 255, 0)  # BGR
            circle_line_thickness = -1  # Thick line and fills in circle
            cv2.circle(frame, (int(kx), int(ky)), size_of_circle, color_of_circle, circle_line_thickness)
    return frame


def draw_edges(frame, keypoints, edges, confidence_threshold):
    """
    draw edges
    :param frame:
    :param keypoints:
    :param edges:
    :param confidence_threshold:
    :return:
    """
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if c1 > confidence_threshold and c2 > confidence_threshold:
            color_of_line = (255, 0, 0)  # BGR
            line_thickness = 2
            # wrapping coordinates so that its just integer
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_of_line, line_thickness)
    return frame
