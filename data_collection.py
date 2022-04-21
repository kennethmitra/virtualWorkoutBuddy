import cv2
import mediapipe as mp
import pandas as pd
import glob
from pathlib import Path

def convert_landmarks_df(results, all_data):
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

    data_folder_name = "data"
    Path(data_folder_name).mkdir(parents=True, exist_ok=True)
    filename = input("Filename: ")
    filepath = f"{data_folder_name}/{filename}.csv"

    columns_fake = [(f"{i}_x", f"{i}_y", f"{i}_z", f"{i}_vis") for i in range(33)]
    columns = []
    for el in columns_fake:
        columns.extend(el)

    df = pd.DataFrame(columns=columns)
    all_data = []

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # # For static images:
    # IMAGE_FILES = []
    # BG_COLOR = (192, 192, 192) # gray
    # with mp_pose.Pose(
    #     static_image_mode=True,
    #     model_complexity=2,
    #     enable_segmentation=True,
    #     min_detection_confidence=0.5) as pose:
    #   for idx, file in enumerate(IMAGE_FILES):
    #     image = cv2.imread(file)
    #     image_height, image_width, _ = image.shape
    #     # Convert the BGR image to RGB before processing.
    #     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #
    #     if not results.pose_landmarks:
    #       continue
    #     print(
    #         f'Nose coordinates: ('
    #         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
    #         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    #     )
    #
    #     annotated_image = image.copy()
    #     # Draw segmentation on the image.
    #     # To improve segmentation around boundaries, consider applying a joint
    #     # bilateral filter to "results.segmentation_mask" with "image".
    #     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    #     bg_image = np.zeros(image.shape, dtype=np.uint8)
    #     bg_image[:] = BG_COLOR
    #     annotated_image = np.where(condition, annotated_image, bg_image)
    #     # Draw pose landmarks on the image.
    #     mp_drawing.draw_landmarks(
    #         annotated_image,
    #         results.pose_landmarks,
    #         mp_pose.POSE_CONNECTIONS,
    #         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    #     # Plot pose world landmarks.
    #     mp_drawing.plot_landmarks(
    #         results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

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

            if cv2.waitKey(3) & 0xFF == ord('z'):
                print("hello")
                convert_landmarks_df(results, all_data)
            else:
                print("bye")
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

    df = pd.DataFrame(data=all_data, columns=columns)
    df.to_csv(filepath)
    print(f"Saving to {filepath}")
