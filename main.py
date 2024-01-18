import time
import cv2
import numpy as np
import tensorflow as tf
import utils as utils
from constants import KEYPOINT_DICT, KEYPOINT_EDGE_INDS_TO_COLOR

model_path = 'model/3.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# connects to webcam (can also pass in video here: 'video.mp4'/play around with 0)
cap = cv2.VideoCapture(0)

# timer that can takes screenshot every X seconds
start_time = time.time()
# set the number of second intervals
interval = 10
# image count
ss_count = 1


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
    print(keypoints_with_scores)

    utils.draw_edges(frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR, 0.4)
    utils.draw_keypoints(frame, keypoints_with_scores, 0.4)

    # whats captured by webcam
    cv2.imshow('MoveNet Lighting', frame)

    # get the key value
    key = cv2.waitKey(10) & 0xFF
    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()