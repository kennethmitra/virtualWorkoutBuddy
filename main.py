import cv2
import mediapipe as mp
import pandas as pd
import glob
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from joblib import dump, load


class_names = ["pushup_up", "pushup_down", "situp_up", "situp_down", "squat_up", "squat_down"]

def process_results(results):
    row_data = []
    for keypoint in range(33):
        try:
            x = results.pose_landmarks.landmark[keypoint].x
            y = results.pose_landmarks.landmark[keypoint].y
            z = results.pose_landmarks.landmark[keypoint].z
            vis = results.pose_landmarks.landmark[keypoint].visibility

            row_data.extend([x, y, z, vis])
        except:
            print("error")
            pass
    return row_data

if __name__ == '__main__':
    data_folder_name = "data"
    output_folder_name = "outputs"

    model = load(f'{output_folder_name}/model.pkl')

    columns_fake = [(f"{i}_x", f"{i}_y", f"{i}_z", f"{i}_vis") for i in range(33)]
    columns = []
    for el in columns_fake:
        columns.extend(el)

    df = pd.DataFrame(columns=columns)
    all_data = []

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose


    # For webcam input:
    cap = cv2.VideoCapture(1)
    with mp_pose.Pose(
            min_detection_confidence=0.05,
            min_tracking_confidence=0.05) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            try:
                processed_data = process_results(results)
                pred = model.predict([processed_data])
                print(f"{class_names[pred[0]]}")
            except Exception as e:
                print("ERROR", e)
                pass


            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(2) & 0xFF == 27:  # ESC
                break
    cap.release()

