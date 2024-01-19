# AI Yoga Coach

AI Yoga Coach is an application employing machine learning to classify and offer feedback on 5 different yoga poses. The 5 poses include downdog, plank, warrior, tree and goddess.

Our goal was to develop an application that enables individuals to engage in yoga practice within the convenience of their homes, utilizing an AI yoga coach to guide and support them throughout their wellness journey.

![yoga_tree](https://github.com/elaine-lai/ai_yoga_coach/assets/90720708/8b1bd28d-580a-49a2-80e4-f32533dc7f02)

We had also deployed a version that was compatible with an edge computing device, the Jetson Nano. 
The animated image above captures the model in action on the Jetson Nano.

![jetsonnano500](https://github.com/elaine-lai/ai_yoga_coach/assets/90720708/24b9260e-a92e-4645-9e39-7d1194b2201d)


## Steps that we took
- Gathered images of varying yoga poses from Kaggle
- Used a Movenet by Tensorflow to gather the 17 keypoints of a body
- Computed angles for specific body parts to discern the pose
- Developed and trained a Support Vector Classifier model utilizing the computed angles for yoga pose classification.
- Deployed the model into Jetson Nano along with a model that is compatible with the laptop

## Getting started
Clone the repo, and then run `main.py`
You may be prompted to install python libraries such as tensorflow, numpy, pandas, cv2, time, pyttsx3.

When the program starts and runs, the application displays the MoveNet keypoints and edges overlaid on your body. It captures movement at 10-second intervals, and after capturing and classifying a pose, it announces feedback using the pyttsx3 Python library. 

#### Pre trained model used: MoveNet
MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body. Learn more here: https://www.tensorflow.org/hub/tutorials/movenet

MoveNet model (singlepost lightining, TensorFLow Lite): https://www.kaggle.com/models/google/movenet/frameworks/tfLite/variations/singlepose-lightning/versions/1?tfhub-redirect=true
