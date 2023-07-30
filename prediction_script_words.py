import cv2
import numpy as np
import os
import mediapipe as mp
import copy
import itertools

from keras.models import load_model
model = load_model("model_norm_15class.h5")
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

SEQUENCE_LENGTH = 20
actions = ['club', 'father', 'fine', 'help', 'hospital', 'howareyou', 'learn',
       'love', 'mother', 'need', 'school', 'sorry', 'thanks',
       'whatisyourname', 'where']

translator = {
    "howareyou": "كيف حالك؟",
    "fine": "جيد",
    "hospital": "مستشفي", 
    "school": "مدرسة", 
    "thanks": "شكرا",
    "sorry": "اسف", 
    "help": "مساعدة", 
    "where": "اين", 
    "learn": "يتعلم", 
    "club": "نادى",
    "father": "اب",
    "mother": "ام",
    "love": "يحب",
    "whatisyourname": "ما اسمك؟",
    "need": "يحتاج",
}

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def extract_keypoints(image, results):
    debug_image = copy.deepcopy(image)
    H, W, _ = debug_image.shape
    landmark_point = []
    pose = np.array([[min(int(res.x * W), W - 1), min(int(res.y * H), H - 1)] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*2)
    lh = np.array([[min(int(res.x * W), W - 1), min(int(res.y * H), H - 1)] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[min(int(res.x * W), W - 1), min(int(res.y * H), H - 1)] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*2)
    return [pose, lh, rh]

def pre_process_keypoints(landmark_list):
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

def is_talking(lm):
    return (lm[20].y <= lm[14].y+0.1 or lm[19].y <= lm[13].y+0.1)

def only_talikng(video_path):
    talking_video = []
    video_reader = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = video_reader.read()
            if not success:
                break

            frame, results = mediapipe_detection(frame, holistic)
            if results.pose_landmarks and is_talking(results.pose_landmarks.landmark):
                talking_video.append(frame)
                
        video_reader.release()
    return talking_video

def frames_extraction(video_path):
    #get only the frame that achieves the is_talking function
    only_talking_frames = only_talikng(video_path)
    
    #expand or shrink video_frames to achieve the SEQUENCE_LENGTH
    ratio = len(only_talking_frames) / SEQUENCE_LENGTH
    frames = [only_talking_frames[int(i*ratio)] for i in range(SEQUENCE_LENGTH)]
    return frames

def predict_video(vid_path):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        talking_vid_frames = frames_extraction(os.path.join(vid_path))
        window = []
        pred_video = []
        for idx, frame in enumerate(talking_vid_frames):

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # extract keypoints
            keypoints = extract_keypoints(image, results)
            keypoints = pre_process_keypoints(keypoints)
            window.append(keypoints)
            
            # draw landmarks
            draw_landmarks(image, results)
            pred_video.append(image)
            print("=", end="")
            
        print(f"> video frames extraction (done).")

        probabilities = model.predict(np.expand_dims(window, axis=0))[0]
        results = {}
        for i in np.argsort(probabilities)[::-1]:
            results[translator[actions[int(i)]]] = f"{probabilities[i] * 100:5.2f}"

    return results


