import json
import sys
import tkinter as tk
from enum import Enum
import beepy
from PIL import ImageTk, Image
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from joblib import load
import threading
import time

"""
This script runs the Virtual Workout Buddy application.
"""

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

###################
#  Construct GUI  #
###################

root = tk.Tk()
root.title("Virtual Workout Buddy")
# Create a frame
app = tk.Frame(root, bg="white")
app.grid()
# Create box for video
video_box = tk.Label(app)
video_box.grid()
# Create box for exercise instructions
instructions_box = tk.Label(app)
instructions_box.grid()

exercise_text = tk.StringVar()
exercise = tk.Label(instructions_box, textvariable=exercise_text, font=("Arial", 25))
exercise.pack()
exercise_count = tk.StringVar()
count = tk.Label(instructions_box, textvariable=exercise_count, font=("Arial", 25))
count.pack()
pred_label = tk.StringVar()
prediction = tk.Label(instructions_box, textvariable=pred_label, font=('Arial', 25))
prediction.pack()

_point_columns_grouped = [(f"{i}_x", f"{i}_y", f"{i}_z", f"{i}_vis") for i in range(33)]
point_columns = []
for el in _point_columns_grouped:
    point_columns.extend(el)

CLASSES = ["pushup_up", "pushup_down", "situp_up", "situp_down", "squat_up", "squat_down"]

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

current_exercise_index = 0
current_exercise_count = 0
current_prediction = ""
crossover_threshold = {'pushup': 0.5, 'squat': 0.6, 'situp': 0.7}  # Crossover threshold specified per exercise
last_state = None


class exerciseState(Enum):
    UP = 0
    DOWN = 1


def convert_results_to_df(results):
    row_data = []
    for keypoint in range(33):
        x = results.pose_landmarks.landmark[keypoint].x
        y = results.pose_landmarks.landmark[keypoint].y
        z = results.pose_landmarks.landmark[keypoint].z
        vis = results.pose_landmarks.landmark[keypoint].visibility

        row_data.extend([x, y, z, vis])

    return row_data


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


def df_to_angles_df(data, compute_ground_angle=True):
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

    return angles_df


def preprocess_pose(results):
    processed_data = convert_results_to_df(results)
    point_df = pd.DataFrame(data=[processed_data], columns=point_columns)
    angles_df = df_to_angles_df(point_df, compute_ground_angle=True)
    combined_df = pd.concat([angles_df, point_df], axis=1)
    return combined_df


def classify_pose(model, results):
    """
    Classify predicted pose into exercise position
    :param model: model to use for inference
    :param results: predicted pose
    :return: probabilities of each class
    """
    try:
        combined_df = preprocess_pose(results)
        pred_probs = model.predict_proba(combined_df.to_numpy())
    except:
        print("No pose detected")
        return None
    return pred_probs[0]


# function for video streaming
last_time = time.time()


def video_stream():
    """
    Updates the webcam image in GUI and calls pose inference and exercise position classification.
    Sets a callback to call itself after a delay
    """
    global last_time
    # Read image from camera
    success, image = cap.read()  # Read image from camera
    if not success:
        print("Empty camera frame")
        return
    image.flags.writeable = False  # To improve performance, optionally mark the image as not writeable to pass by reference.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract Pose
    pose_results = extract_pose(image)

    # Classify pose into exercise position
    position_probs = classify_pose(model, pose_results)

    # Update GUI based on classification results
    process_classification(position_probs)

    # Draw pose on image
    image = draw_pose(image, pose_results)

    # Add image to box
    image = cv2.flip(image, 1)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    video_box.imgtk = imgtk
    video_box.configure(image=imgtk)

    updateInstructions()
    check_load_next_exercise()

    # Calculate FPS
    print("Current FPS:", 1 / (time.time() - last_time))
    last_time = time.time()

    video_box.after(15, video_stream)


def process_classification(position_probs):
    """
    Determine if rep counter should be increased
    :param position_probs:
    :return:
    """
    global current_exercise_index
    global current_exercise_count
    global current_prediction
    global last_state

    if position_probs is None:
        return

    exercise_name = exercises[current_exercise_index]['name']
    ex_up_idx = CLASSES.index(f"{exercise_name}_up")
    ex_down_idx = CLASSES.index(f"{exercise_name}_down")

    # Extract probabilities of the up and down positions for the current exercise
    p_up = position_probs[ex_up_idx]
    p_down = position_probs[ex_down_idx]

    # Normalize p_up, p_down to sum to 1.0
    p_up, p_down = (p_up / (p_up + p_down), p_down / (p_up + p_down))
    current_state = last_state

    # Determine if exercise state should be changed
    if p_up > crossover_threshold[exercise_name] and p_up > p_down:
        current_state = exerciseState.UP
    elif p_down > crossover_threshold[exercise_name] and p_down > p_up:
        current_state = exerciseState.DOWN

    # Determine if rep count should increase
    if current_state == exerciseState.UP and last_state == exerciseState.DOWN:
        current_exercise_count += 1
        # Provide visual feedback
        app['bg'] = 'green'
        # Provide audio feedback
        beepSound("coin")
    else:
        # Provide visual feedback
        app['bg'] = 'white'

    current_prediction = (CLASSES[np.array(position_probs).argmax()])
    last_state = current_state


def updateInstructions():
    """
    Update textboxes on GUI
    """
    if current_exercise_index < len(exercises):
        exercise_text.set(f"{exercises[current_exercise_index]['name']}s")
        exercise_count.set(f"{current_exercise_count}/{exercises[current_exercise_index]['count']}")
    pred_label.set(f"Predicted Label: {current_prediction}")


def extract_pose(image):
    # Extract Pose
    results = pose.process(image)
    return results


def draw_pose(image, pose_results):
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    mp_drawing.draw_landmarks(
        image,
        pose_results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    return image


def check_load_next_exercise():
    """
    Check if the program should move on to the next exercise on the list
    :return:
    """
    global current_exercise_count
    global current_exercise_index
    global last_state

    if current_exercise_count >= exercises[current_exercise_index]['count']:
        current_exercise_index += 1
        current_exercise_count = 0
        last_state = None

        updateInstructions()
        beepSound("ready")

        if current_exercise_index >= len(exercises):
            beepSound("success")
            print("Workout Complete!")
            exercise_text.set("Workout Complete!")
            sys.exit(0)


def beepSound(name):
    """
    Play sound in a new thread so the GUI does not freeze
    """
    thread = threading.Thread(target=lambda: beepy.beep(name))
    thread.start()


if __name__ == '__main__':
    # Create Pose model
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=0,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        smooth_segmentation=True,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.3)

    # Load model
    model = load('outputs/model.pkl')

    # Capture from camera
    cap = cv2.VideoCapture(0)

    # Read exercise list
    with open('config.json', 'r') as f:
        exercises = json.load(f)

    check_load_next_exercise()

    print("Starting...")

    # Run GUI
    video_stream()
    root.mainloop()
