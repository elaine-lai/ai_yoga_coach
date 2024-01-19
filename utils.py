import numpy as np
import cv2
import pandas as pd

from constants import KEYPOINT_DICT

# keypoints = [[0.3923125, 0.5542421, 0.5562568],
#              [0.3295156, 0.60111874, 0.58333707],
#              [0.34902796, 0.5428777, 0.54064333],
#              [0.3278386, 0.74546766, 0.5753689],
#              [0.34782788, 0.60988575, 0.5702256],
#              [0.5114173, 0.95146173, 0.6981182],
#              [0.5708685, 0.59803444, 0.7739016],
#              [0.8679228, 0.8845679, 0.27316803],
#              [0.83263266, 0.46695566, 0.5313083],
#              [0.6359774, 0.65353507, 0.01975938],
#              [0.80994785, 0.3590023, 0.12591188],
#              [0.9310886, 0.9022478, 0.0241341],
#              [0.9304087, 0.56741756, 0.02165917],
#              [0.82178545, 0.7025744, 0.02470911],
#              [0.8333831, 0.4202083, 0.05160204],
#              [0.85987794, 0.28691214, 0.01190803],
#              [0.8683524, 0.24162577, 0.02748439]]

# [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]).


def check_threshold(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    # print(type(keypoints))
    # print(keypoints)
    condition = keypoints[:, :, :, -1] > 0.1
    # Use boolean indexing to filter elements
    filtered_keypoints = keypoints[condition]
    # print(len(filtered_keypoints))
    # print(filtered_keypoints)
    # print("\n\n")
    return len(filtered_keypoints)


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

# Maps keypoints for angle calculations
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

OUTPUT_FOLDER = 'csv_outputs/'


# def keypoints_by_directory(movenet, directory):
#     """Used for processing keypoints for images in a directory"""
#     kps = []
#     # Gets keypoints from each image in directory
#     for index, filename in enumerate(os.listdir(directory)):
#         if not filename.endswith(".DS_Store"):
#             filepath = os.path.join(directory, filename)
#             print(filepath)
#             print(index)
#             print((index / len(os.listdir(directory))) * 100, '% complete')
#             # image = movenet.process_image(filepath)
#             keypoints_with_score = movenet.get_keypoints_with_scores(filepath)
#             kps.append(keypoints_with_score)
#     return kps


def map_coordinates_to_keypoint(keypoints, output_csv=False):
    """X and y coordinates are mapped to their respective keypoint"""
    # Filter keypoint coordinates(xy) without confidence levels
    if isinstance(keypoints, list):
        filtered_kps = [[values[:2] for values in kp.tolist()]
                        for kp in keypoints]
    else:
        filtered_kps = [[values[:2] for values in keypoints.tolist()]]
    # Map the x and y values to their keypoints
    mapped_kp_coords = []
    for xy in filtered_kps:
        row = {}
        for i, keypoint in enumerate(KEYPOINT_DICT.keys()):
            row[f"{keypoint}_x"] = xy[i][0]
            row[f"{keypoint}_y"] = xy[i][1]
        mapped_kp_coords.append(row)
    mapped_kp_coords_df = pd.DataFrame(mapped_kp_coords)
    if output_csv:
        coordinate_filename = f"{OUTPUT_FOLDER}keypoint_coordinates.csv"
        mapped_kp_coords_df.to_csv(coordinate_filename, index=False)
    return mapped_kp_coords_df


def get_angles(mapped_kp_coords, append=False, output_csv=False):
    """Gets the angles for all rows(snapshots) in a dataframe. Set append to True if original keypoint coordinates
    should be retained in the resulting DataFrame."""
    # Apply the calculate_angle function to each row of the DataFrame
    angle_series = mapped_kp_coords.apply(
        calculate_angles, axis=1, angle_dict=ANGLE_DICT)
    angle_df = pd.DataFrame.from_dict(angle_series.to_dict(), orient='index')
    # Append the angles to the original DataFrame
    if append:
        angle_df = pd.concat([mapped_kp_coords, angle_df], axis=1)
    if output_csv:
        angle_df.to_csv(
            f"{OUTPUT_FOLDER}keypoints_with_angles.csv", index=False)
    return angle_df


def tuplize_coordinates(row, keypoint):
    """Helper function for combining xy coordinates"""
    x_col = f"{keypoint[0]}_x"
    y_col = f"{keypoint[0]}_y"
    return [row[x_col], row[y_col]]


def calculate_angles(row, angle_dict):
    """Helper function to calculate the angles for a row(snapshot)"""
    angles = {}
    for angle_key, keypoints in angle_dict.items():
        keypoint_coords = [tuplize_coordinates(
            row, [keypoint]) for keypoint in keypoints]
        angle = calculate_angle_from_coords(*keypoint_coords)
        angles[angle_key] = angle
    return angles


def calculate_angle_from_coords(coord1, coord2, coord3):
    """Angle calculation from vectors"""
    vector1 = np.array(coord1) - np.array(coord2)
    vector3 = np.array(coord3) - np.array(coord2)

    dot_product = np.dot(vector1, vector3)
    magnitude1 = np.linalg.norm(vector1)
    magnitude3 = np.linalg.norm(vector3)

    cos_angle = dot_product / (magnitude1 * magnitude3)
    # Ensure the cosine value is within the valid range
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # Calculate the angle in radians
    angle_rad = np.arccos(cos_angle)
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def process_keypoints_to_angles(keypoints, print_result=True):
    kp_coordinates = map_coordinates_to_keypoint(keypoints)
    angles_df = get_angles(kp_coordinates)
    if print_result:
        print(f"Angles:\n{angles_df.head()}")
    return angles_df


def predict_class(model, angles, print_result=True):
    # model = joblib.load('models/svc_model.sav')
    # pred_label = model.predict(angles)  # Temp replaced with pose that is max(prob)
    pred_probs = model.predict_proba(angles)
    index_of_max = np.argmax(pred_probs, axis=1)[0]
    # Get the class labels
    class_labels = model.classes_
    pred_label = np.array([class_labels[index_of_max]]
                          )  # Replacing model.predict()
    # Create a DataFrame for better visualization
    prob_df = pd.DataFrame(pred_probs, columns=[
                           f'{label}(%)' for label in class_labels])
    # Print the probabilities with class labels
    result_df = pd.concat(
        [pd.DataFrame({'True Label': pred_label}), prob_df], axis=1)
    if print_result:
        print(f"Class probabilities:\n{result_df.head()}")
    return result_df


# Create the output folder if it doesn't exist
# if not os.path.exists(OUTPUT_FOLDER):
#     os.makedirs(OUTPUT_FOLDER)
#     print(f"Folder '{OUTPUT_FOLDER}' created.")