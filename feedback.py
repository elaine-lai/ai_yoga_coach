import pandas as pd

RANGES = {
    "downdog": [],
    "goddess": [],
    "plank": [],
    "tree": [],
    "warrior": []
}


def writeToFile(data, filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # df = pd.DataFrame()
        new_row = pd.DataFrame([data])
        new_row.to_csv(filepath, index=False, header=False)
    else:
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filepath, index=False, header=False)


def evaluateDowndog(angles, keypoints):
    # writeToFile(angles, 'downdog_angles.csv')
    # writeToFile(keypoints, 'downdog_keypoints.csv')

    feedback = []
    feedback_reasons = {}

    # L and R elbow are too bent
    if angles[0] < RANGES['downdog'][0][0] and angles[4] < RANGES['downdog'][4][0]:
        feedback_reasons['Push through your palms and straighten both elbows'] = (RANGES['downdog'][0][0] - angles[0], RANGES['downdog'][4][0] - angles[4])
        feedback.append('Push through your palms and straighten both elbows')
    # left elbow is too bent
    elif angles[0] < RANGES['downdog'][0][0]:
        feedback_reasons['Straighten left elbow'] = RANGES['downdog'][0][0] - angles[0]
        feedback.append('Straighten left elbow')
    # right elbow is too bent
    elif angles[4] < RANGES['downdog'][4][0]:
        feedback_reasons['Straighten right elbow'] = RANGES['downdog'][4][0] - angles[4]
        feedback.append('Straighten right elbow')

    # L and R hip are too saggy
    if angles[2] > RANGES['downdog'][2][1] and angles[6] > RANGES['downdog'][6][1]:
        feedback_reasons['Push through your palms and feet to raise your hips'] = (angles[2] - RANGES['downdog'][2][0], angles[6] - RANGES['downdog'][6][0])
        feedback.append('Push through your palms and feet to raise your hips')  # raise butt higher?? butt is sagging??

    # L and R knee are too bent
    if angles[3] < RANGES['downdog'][3][0] and angles[7] < RANGES['downdog'][7][0]:
        feedback_reasons['Push through your feet to straighten both knees'] = (RANGES['downdog'][3][0] - angles[3], RANGES['downdog'][7][0] - angles[7])
        feedback.append('Push through your feet to straighten both knees')
    # left knee
    elif angles[3] < RANGES['downdog'][3][0]:
        feedback_reasons['Straighten left knee'] = RANGES['downdog'][3][0] - angles[3]
        feedback.append('Straighten left knee')
    # right knee
    elif angles[7] < RANGES['downdog'][7][0]:
        feedback_reasons['Straighten right knee'] = RANGES['downdog'][7][0] - angles[7]
        feedback.append('Straighten right knee')

    return feedback, feedback_reasons


def evaluateGoddess(angles, keypoints):
    # writeToFile(angles, 'goddess_angles.csv')
    # writeToFile(keypoints, 'goddess_keypoints.csv')

    feedback = []
    feedback_reasons = {}

    # L and R wrists are below elbows
    if keypoints[8] < keypoints[4] and keypoints[10] < keypoints[6]:
        feedback_reasons['Raise both wrists above elbows'] = (keypoints[4] - keypoints[8], keypoints[6] - keypoints[10])
        feedback.append('Raise both wrists above elbows')
    # L wrist is below L elbow
    elif keypoints[8] < keypoints[4]:
        feedback_reasons['Raise left wrist above left elbow'] = keypoints[4] - keypoints[8]
        feedback.append('Raise left wrist above left elbow')
    # R wrist is below R elbow
    elif keypoints[10] < keypoints[6]:
        feedback_reasons['Raise right wrist above right elbow'] = keypoints[6] - keypoints[10]
        feedback.append('Raise right wrist above right elbow')

    # L and R elbow are too straight
    if angles[0] > RANGES['goddess'][0][1] and angles[4] > RANGES['goddess'][4][1]:
        feedback_reasons['Bend both elbows'] = (angles[0] - RANGES['goddess'][0][0], angles[4] - RANGES['goddess'][4][0])
        feedback.append('Bend both elbows')
    # L elbow is too straight
    elif angles[0] > RANGES['goddess'][0][1]:
        feedback_reasons['Bend left elbow'] = angles[0] - RANGES['goddess'][0][0]
        feedback.append('Bend left elbow')
    # R elbow is too straight
    elif angles[4] > RANGES['goddess'][4][1]:
        feedback_reasons['Bend right elbow'] = angles[4] - RANGES['goddess'][4][0]
        feedback.append('Bend right elbow')

    # L and R armpit are not square ie below q1
    if angles[1] < RANGES['goddess'][1][0] and angles[5] < RANGES['goddess'][5][0]:
        feedback_reasons['Raise both arms'] = (RANGES['goddess'][1][0] - angles[1], RANGES['goddess'][5][0] - angles[5])
        feedback.append('Raise both arms')

    # L and R knee are not bent, ie below above q3
    if angles[3] > RANGES['goddess'][3][1] and angles[7] > RANGES['goddess'][7][1]:
        feedback_reasons['Bend both knees'] = (angles[3] - RANGES['goddess'][3][1], angles[7] - RANGES['goddess'][7][1])
        feedback.append('Bend both knees')

    return feedback, feedback_reasons


def evaluatePlank(angles, keypoints):
    # writeToFile(angles, 'plank_angles.csv')
    # writeToFile(keypoints, 'plank_keypoints.csv')

    feedback = []
    feedback_reasons = {}

    # L and R elbow are bent
    if angles[0] < RANGES['plank'][0][0] and angles[4] < RANGES['plank'][4][0]:
        feedback_reasons['Straighten both elbows'] = (RANGES['plank'][0][0] - angles[0], RANGES['plank'][4][0] - angles[4])
        feedback.append('Straighten both elbows')
    # left elbow is bent
    elif angles[0] < RANGES['plank'][0][0]:
        feedback_reasons['Straighten left elbow'] = RANGES['plank'][0][0] - angles[0]
        feedback.append('Straighten left elbow')
    # right elbow is bent
    elif angles[4] < RANGES['plank'][4][0]:
        feedback_reasons['Straighten right elbow'] = RANGES['plank'][4][0] - angles[4]
        feedback.append('Straighten right elbow')

    # L and R shoulder are not square ie above q3 (too straight)
    if angles[1] > RANGES['plank'][1][1] and angles[5] > RANGES['plank'][5][1]:
        feedback_reasons['Square shoulders'] = (angles[1] - RANGES['plank'][1][1], angles[5] - RANGES['plank'][5][1])
        feedback.append('Square shoulders')

    # L and R hip are not square ie below q1 (too bent)
    if angles[2] < RANGES['plank'][2][0] and angles[6] < RANGES['plank'][6][0]:
        feedback_reasons['Keep hips square and level'] = (RANGES['plank'][2][0] - angles[2], RANGES['plank'][6][0] - angles[6])
        feedback.append('Keep hips square and level')

    # L and R knee are too bent
    if angles[3] < RANGES['plank'][3][0] and angles[7] < RANGES['plank'][7][0]:
        feedback_reasons['Straighten both knees'] = (RANGES['plank'][3][0] - angles[3], RANGES['plank'][7][0] - angles[7])
        feedback.append('Straighten both knees')
    # left knee
    elif angles[3] < RANGES['plank'][3][0]:
        feedback_reasons['Straighten left knee'] = RANGES['plank'][3][0] - angles[3]
        feedback.append('Straighten left knee')
    # right knee
    elif angles[7] < RANGES['plank'][7][0]:
        feedback_reasons['Straighten right knee'] = RANGES['plank'][7][0] - angles[7]
        feedback.append('Straighten right knee')

    return feedback, feedback_reasons


def evaluateTree(angles, keypoints):
    # writeToFile(angles, 'tree_angles.csv')
    # writeToFile(keypoints, 'tree_keypoints.csv')

    feedback = []
    feedback_reasons = {}

    # L and R wrist are below shoulders
    if keypoints[8] < keypoints[0] and keypoints[10] < keypoints[2]:
        feedback_reasons['Shoot your arms to the sky'] = (keypoints[5] - keypoints[9], keypoints[7] - keypoints[11])
        feedback.append('Shoot your arms to the sky')
    # L wrist is below L shoulder
    elif keypoints[8] < keypoints[0]:
        feedback_reasons['Raise left arm'] = keypoints[0] - keypoints[8]
        feedback.append('Raise left arm')
    # R wrist is below R shoulder
    elif keypoints[10] < keypoints[2]:
        feedback_reasons['Raise right arm'] = keypoints[2] - keypoints[10]
        feedback.append('Raise right arm')

    # L and R elbow are too bent
    if angles[0] < RANGES['tree'][0][0] and angles[4] < RANGES['tree'][4][0]:
        feedback_reasons['Straighten both elbows'] = (RANGES['tree'][0][0] - angles[0], RANGES['tree'][4][0] - angles[4])
        feedback.append('Straighten both elbows')

    # L and R knee - both knees are bent OR both knees are straight
    if (angles[3] < RANGES['tree'][3][0] and angles[7] < RANGES['tree'][7][0]) or \
            (angles[3] > RANGES['tree'][3][1] and angles[7] > RANGES['tree'][7][1]):
        feedback_reasons['Straighten one knee and bend the other1'] = (RANGES['tree'][3][0] - angles[3], RANGES['tree'][7][0] - angles[7])
        feedback_reasons['Straighten one knee and bend the other2'] = (angles[3] - RANGES['tree'][3][1], angles[7] - RANGES['tree'][7][1])
        feedback.append('Stand all your weight on one leg and bend the other')

    return feedback, feedback_reasons


def evaluateWarrior(angles, keypoints):
    # writeToFile(angles, 'warrior_angles.csv')
    # writeToFile(keypoints, 'warrior_keypoints.csv')

    feedback = []
    feedback_reasons = {}

    # L and R elbow are too bent
    if angles[0] < RANGES['warrior'][0][0] and angles[4] < RANGES['warrior'][4][0]:
        feedback_reasons['Extend your arms outwards to straighten both elbows'] = (RANGES['warrior'][0][0] - angles[0], RANGES['warrior'][4][0] - angles[4])
        feedback.append('Extend your arms outwards to straighten both elbows')
    # left elbow
    elif angles[0] < RANGES['warrior'][0][0]:
        feedback_reasons['Straighten left elbow'] = RANGES['warrior'][0][0] - angles[0]
        feedback.append('Straighten left elbow')
    # right elbow
    elif angles[4] < RANGES['warrior'][4][0]:
        feedback_reasons['Straighten right elbow'] = RANGES['warrior'][4][0] - angles[4]
        feedback.append('Straighten right elbow')

    # L and R shoulder should be square
    if angles[1] > RANGES['warrior'][1][1] and angles[5] > RANGES['warrior'][5][1]:
        feedback_reasons['Relax and square your shoulders'] = (angles[1] - RANGES['warrior'][1][1], angles[5] - RANGES['warrior'][5][1])
        feedback.append('Relax and square your shoulders')

    # L and R arms are too low
    if angles[1] < RANGES['warrior'][1][0] and angles[5] < RANGES['warrior'][5][0]:
        feedback_reasons['Engage your core and level both arms'] = (keypoints[0] - keypoints[8], keypoints[2] - keypoints[10])
        feedback.append('Engage your core and level both arms')

    # L and R knee - both knees are straight or both knees are bent
    if (angles[3] < RANGES['warrior'][3][0] and angles[7] < RANGES['warrior'][7][0]) or \
            (angles[3] > RANGES['warrior'][3][1] and angles[7] > RANGES['warrior'][7][1]):
        feedback_reasons['Straighten one knee and bend the other1'] = (RANGES['warrior'][3][0] - angles[3], angles[7] - RANGES['warrior'][7][1])
        feedback_reasons['Straighten one knee and bend the other2'] = (angles[3] - RANGES['warrior'][3][1], RANGES['warrior'][7][0] - angles[7])
        feedback.append('Lean forward into a slight lunge')

    return feedback, feedback_reasons


FEEDBACK_FUNCS = {
    "downdog": evaluateDowndog,
    "goddess": evaluateGoddess,
    "plank": evaluatePlank,
    "tree": evaluateTree,
    "warrior": evaluateWarrior
}


def setRanges():
    df = pd.read_csv('data/pose_ranges.csv', header=[0, 1])
    poses = ['downdog', 'goddess', 'plank', 'tree', 'warrior']
    for index, pose in enumerate(poses):
        pose_stats = df.iloc[index].tolist()
        pose_stats = [(pose_stats[i], pose_stats[i + 1]) for i in range(0, len(pose_stats), 2)]
        RANGES[pose] = pose_stats


def preprocess_angles(angles):
    df = pd.DataFrame(angles)
    angles_list = df.values.flatten().tolist()
    return angles_list


def preprocess_keypoints(keypoints):
    keypoint_list = keypoints.tolist()
    modified_keypoints_list = [item for sublist in keypoint_list for index, item in enumerate(sublist) if index != 2]
    return modified_keypoints_list


def evaluatePose(pose, angles, keypoints):
    # print(pose)
    angles = preprocess_angles(angles)
    # print(angles)
    keypoints = preprocess_keypoints(keypoints)
    # print(keypoints)
    setRanges()
    feedback, feedback_reasons = FEEDBACK_FUNCS[pose](angles, keypoints)
    # print(feedback)
    return feedback, feedback_reasons