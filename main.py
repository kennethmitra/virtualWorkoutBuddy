import cv2
import mediapipe as mp
import numpy as np
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


LANDMARK_MODEL = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2,
    'left_eye_outer': 3, 'right_eye_inner': 4, 'right_eye': 5,
    'right_eye_outer': 6, 'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10, 'left_shoulder': 11,
    'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17,
    'right_pinky': 18, 'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23,
    'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29,
    'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32
}


def calculate_angle(p1, vertex, p2):
    vector1 = vertex - p1
    vector2 = vertex - p2
    v1_norm = np.linalg.norm(vector1, axis=1)
    v2_norm = np.linalg.norm(vector2, axis=1)
    angle = np.arccos(np.sum(vector1 * vector2, axis=1) / (v1_norm * v2_norm))
    return angle / np.pi * 180


def compute_angle_to_ground_normal(p1, p2):
    vector1 = p1 - p2
    vector2 = np.array([0, 1, 0])
    return np.sum(vector1 * vector2, axis=1) / (np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2)) / np.pi * 180


def get_point(df, landmark):
    landmark_index = LANDMARK_MODEL[landmark]

    return df[[f'{landmark_index}_x', f'{landmark_index}_y', f'{landmark_index}_z']].values


def generate_angle_feature(df, out, landmark_a, landmark_b, landmark_c):
    out[f'{landmark_a}_to_{landmark_b}_to_{landmark_c}'] = calculate_angle(get_point(df, landmark_a),
                                                                           get_point(df, landmark_b),
                                                                           get_point(df, landmark_c))


def df_to_angles_df(data, compute_ground_angle=False):
    angles_df = pd.DataFrame()
    generate_angle_feature(data, angles_df, 'left_index', 'left_wrist', 'left_elbow')
    generate_angle_feature(data, angles_df, 'right_index', 'right_wrist', 'right_elbow')
    generate_angle_feature(data, angles_df, 'left_wrist', 'left_elbow', 'left_shoulder')
    generate_angle_feature(data, angles_df, 'right_wrist', 'right_elbow', 'right_shoulder')
    generate_angle_feature(data, angles_df, 'left_elbow', 'left_shoulder', 'left_hip')
    generate_angle_feature(data, angles_df, 'right_elbow', 'right_shoulder', 'right_hip')
    generate_angle_feature(data, angles_df, 'left_elbow', 'left_shoulder', 'right_shoulder')
    generate_angle_feature(data, angles_df, 'right_elbow', 'right_shoulder', 'left_shoulder')
    generate_angle_feature(data, angles_df, 'left_shoulder', 'left_hip', 'left_knee')
    generate_angle_feature(data, angles_df, 'right_shoulder', 'right_hip', 'right_knee')
    generate_angle_feature(data, angles_df, 'left_hip', 'left_knee', 'left_ankle')
    generate_angle_feature(data, angles_df, 'right_hip', 'right_knee', 'right_ankle')
    generate_angle_feature(data, angles_df, 'left_knee', 'left_hip', 'right_hip')
    generate_angle_feature(data, angles_df, 'right_knee', 'right_hip', 'left_hip')
    generate_angle_feature(data, angles_df, 'left_foot_index', 'left_ankle', 'left_knee')
    generate_angle_feature(data, angles_df, 'right_foot_index', 'right_ankle', 'right_knee')

    if compute_ground_angle:
        neck = (get_point(data, 'right_shoulder') + get_point(data, 'left_shoulder')) / 2
        pelvis = (get_point(data, 'right_hip') + get_point(data, 'left_hip')) / 2
        angles_df['neck_to_pelvis_to_ground'] = compute_angle_to_ground_normal(neck, pelvis)

    angles_df['label'] = data['label']
    return angles_df


if __name__ == '__main__':

    # Load Model
    model = load(f'outputs/model.pkl')

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(1)
    with mp_pose.Pose(min_detection_confidence=0.50, min_tracking_confidence=0.50) as pose:
        while cap.isOpened():
            success, image = cap.read()  # Read image from camera
            if not success:
                print("Empty camera frame")
                continue

            # Extract Pose
            image.flags.writeable = False  # To improve performance, optionally mark the image as not writeable to pass by reference.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Classify Pose
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
