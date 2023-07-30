import cv2
import numpy as np
import mediapipe as mp
import copy
import itertools

from keras.models import load_model
model = load_model("ann_model.h5")

mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True,
                      max_num_hands=1,
                      min_detection_confidence=0.5, 
                      min_tracking_confidence=0.5)

actions = ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = hands.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    for handLms in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def predict_number(path):
    image = cv2.imread(path)
    # image = cv2.flip(image, 1)  
    debug_image = copy.deepcopy(image)
    image, results = mediapipe_detection(image, hands)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            prediction = model.predict(np.array([pre_processed_landmark_list]))[0]
        

    return {"result_number": str(np.argmax(prediction))}