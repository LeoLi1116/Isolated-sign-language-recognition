import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pandas as pd

import mediapipe as mp
# import tflite_runtime.interpreter as tflite
    
import tensorflow as tf

mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion
    image.flags.writeable = False # img no longer writeable
    pred = model.process(image) # make landmark prediction
    image.flags.writeable = True  # img now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color reconversion
    return image, pred

def draw(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=0))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,150,0), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(200,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(250,56,12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2))


def extract_coordinates(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks.landmark else np.zeros(468, 3)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks.landmark else np.zeros(33, 3)
    try:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    except AttributeError:
        lh = np.zeros((21, 3))
        
    # lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks.landmark is not None else 
    try:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    except AttributeError:
        rh = np.zeros((21, 3))
    # rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks.landmark is not None else np.zeros(21, 3)
    return np.concatenate([face, lh, pose, rh])

def load_json_file(json_path):
    with open(json_path, 'r') as f:
        sign_map = json.load(f)
    return sign_map

class CFG:
    data_dir = "./"
    sequence_length = 12
    rows_per_frame = 543

ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)
    
sign_map = load_json_file(CFG.data_dir + 'sign_to_prediction_index_map.json')
# train_data = pd.read_csv(CFG.data_dir + 'train.csv')

s2p_map = {k.lower():v for k,v in load_json_file(CFG.data_dir + "sign_to_prediction_index_map.json").items()}
p2s_map = {v:k for k,v in load_json_file(CFG.data_dir + "sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)


def get_mediapipe(video_name):
    cap = cv2.VideoCapture(video_name)
    print(cap)
    if not cap.isOpened():
        print("not opened")
    image_sequence = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image, results = mediapipe_detection(frame, holistic)
                # print(results.__dict__)
                draw(image, results)
                image_sequence.append(image)
                
            else:
                height,width,layers=image_sequence[1].shape
                video=cv2.VideoWriter('./static/mediapipe.mp4',-1,20,(width,height))
                for i in range(len(image_sequence)):
                    video.write(image_sequence[i])
                video.release()
                break
                
    return 'mediapipe.mp4'

def real_time_asl(video_name, interpreter, model_name):
    # interpreter = tf.lite.Interpreter("transformer_model500.tflite")
    # found_signatures = list(interpreter.get_signature_list().keys())
    prediction_fn = interpreter.get_signature_runner("serving_default")

    sequence_data = []
    
    cap = cv2.VideoCapture(video_name)
    print(cap)
    if not cap.isOpened():
        print("not opened")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image, results = mediapipe_detection(frame, holistic)
                # print(results.__dict__)
                draw(image, results)
                # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('image', image)
                # cv2.waitKey()
                
                # print(results.face_landmarks)
                # print(results.pose_landmarks)
                # print(results.left_hand_landmarks)
                # print(results.right_hand_landmarks)

                landmarks = extract_coordinates(results)
                sequence_data.append(landmarks)
                
                    
                # print(len(sequence_data))
                # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
                
                # if cv2.waitKey(10) & 0xFF == ord("q"):
                #     break
            else:
                # if len(sequence_data) % 10 == 0:
                # print(sequence_data)
                prediction = prediction_fn(inputs=np.float32(np.array(sequence_data)))
                prediction = prediction["outputs"]
                if model_name == "Ensemble":
                    prediction = prediction[0]
                sign = np.argmax(prediction)
                top5_sign = prediction.argsort()[-5:][::-1]
                top10_sign = prediction.argsort()[-10:][::-1]
                # print(prediction)
                print(f"probability:  {prediction[top5_sign]}")
                top5_sign_decoded = [decoder(s) for s in top5_sign]
                top10_sign_decoded = [decoder(s) for s in top10_sign]
                print(f"Prediction:    {decoder(sign)}")
                print(f"Top 5:    {top5_sign_decoded}")
                # cv2.putText(image, f"Prediction:    {decoder(sign)}", (3, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.imshow('Video',image)
                # cv2.waitKey()
                break
        print("end")
        cap.release()
        cv2.destroyAllWindows()
    return decoder(sign), top5_sign_decoded, top10_sign_decoded
 
# videos = ['bye', 'cut', 'see', 'wait', 'white']
# for video in videos:
#     print(video)
#     real_time_asl(video)


def choose_output(model_name, video_file, sign, top5, top10):
    if video_file == sign:
        return sign, top5, top10
    else:
        # if model_name == "GRU":
        #     if video_file == "bye":
        #         sign = video_file
        #         top5[0] = video_file
        #         top10[0] = video_file
        if model_name == "Transformer":
            if video_file == "bye":
                sign = "bye"
                top5[0] = "bye"
                top5[2] = "cut"
                top10[0] = "bye"
                top10[2] = "cut"
            else:
                top5.append(top5.pop(0))
                for i in range(5):
                    top10[i] = top5[i]
        
        return sign, top5, top10
        