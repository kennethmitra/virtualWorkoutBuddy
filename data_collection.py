import cv2
import mediapipe as mp
import pandas as pd
import glob
from pathlib import Path


def convert_landmarks_df(results, all_data):
    """
    Converts BlazePose output to a list
    :param results: BlazePose output
    :param all_data: list to append to
    :return: None
    """
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
    if len(row_data) > 0:
        all_data.append(row_data)


if __name__ == '__main__':

    # Create data directory
    data_folder_name = "data"
    Path(data_folder_name).mkdir(parents=True, exist_ok=True)
    filename = input("Filename: ")
    filepath = f"{data_folder_name}/{filename}.csv"

    # Column names for DataFrame
    columns_fake = [(f"{i}_x", f"{i}_y", f"{i}_z", f"{i}_vis") for i in range(33)]
    columns = []
    for el in columns_fake:
        columns.extend(el)

    all_data = []  # List to hold collected data

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)  # Open webcam stream. MODIFY this if you need to use a different webcam
    with mp_pose.Pose(min_detection_confidence=0.05, min_tracking_confidence=0.05) as pose:
        while cap.isOpened():
            success, image = cap.read()  # Get image
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Compute Pose
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # If z key is pressed, save data
            if cv2.waitKey(3) & 0xFF == ord('z'):
                print("saving")
                convert_landmarks_df(results, all_data)

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

    # Save collected data as CSV
    df = pd.DataFrame(data=all_data, columns=columns)
    df.to_csv(filepath)
    print(f"Saving to {filepath}")
