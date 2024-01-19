import time
import cv2
import joblib
import numpy as np
import tensorflow as tf
# import utils as utils
from utils import check_threshold, draw_edges, draw_keypoints, process_keypoints_to_angles, predict_class
from constants import KEYPOINT_EDGE_INDS_TO_COLOR
from feedback import evaluatePose

model_path = 'model/3.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# connects to webcam (can also pass in video here: 'video.mp4'/play around with 0)
cap = cv2.VideoCapture(0)

# timer that can takes screenshot every X seconds
start_time = time.time()
# set the number of second intervals
interval = 5
# image count
ss_count = 1

# model
model = joblib.load('model/svc_model.joblib')
# confidence threshold
confidence_threshold = 0.1

# loop through every single frame in webcam
while cap.isOpened():
    # ret = whether a frame was successfully read or not (success status)
    # frame = frame that was read, image represented as arrays [480H, 640W, 3channels]
    ret, frame = cap.read()

    # reshape image, movenet only takes in 192x192
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    # setup input and output (part of working with tflite)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # gets the input_details, and then sets the 'index' key to be the input_image
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    # make our predictions
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # draw the keypoints and edges
    draw_edges(frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR, 0.1)
    draw_keypoints(frame, keypoints_with_scores, 0.1)
    number_of_keypoints_pass_threshold = check_threshold(frame, keypoints_with_scores, 0.1)

    # show the webcam by opencv
    cv2.imshow('MoveNet Lighting', frame)
    # timer
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= interval:
        trimmed_keypoints_with_scores = keypoints_with_scores[0][0]
        if number_of_keypoints_pass_threshold == 17:
            angles = process_keypoints_to_angles(trimmed_keypoints_with_scores)
            model_probabilities = predict_class(model, angles)  # gives a dataframe with all probabilities
            predicted_label = model_probabilities['True Label'].iloc[0]  # gets the predicted model label
            feedback, feedback_reasons = evaluatePose(predicted_label, angles, trimmed_keypoints_with_scores)
        else:
            print("====================================")
            print("Not enough keypoints, please make sure you are in frame!")
            print("====================================\n")
        start_time = time.time()

    # get the key value
    key = cv2.waitKey(10) & 0xFF
    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
