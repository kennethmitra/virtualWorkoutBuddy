import autosklearn.classification
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load

# Create dataset
class_names = ["pushup_up", "pushup_down", "situp_up", "situp_down", "squat_up", "squat_down"]


# Create dataset
def read_dataset(class_names):
    file_names = [f"data/{name}.csv" for name in class_names]
    test_file_names = [f"data/test_{name}.csv" for name in class_names]
    per_class_dfs = []
    per_class_test_dfs = []

    for i, class_name in enumerate(class_names):
        df = pd.read_csv(file_names[i], index_col=0)
        test_df = pd.read_csv(test_file_names[i], index_col=0)
        df['label'] = i
        test_df['label'] = i
        df = df.iloc[::1, :]
        test_df = test_df.iloc[::1, :]
        per_class_dfs.append(df)
        per_class_test_dfs.append(test_df)
        print(f"{class_name} - {len(df)}")

    train = pd.concat(per_class_dfs, ignore_index=True)
    test = pd.concat(per_class_test_dfs, ignore_index=True)

    return train, test


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


def get_point(df, landmark):
    landmark_index = LANDMARK_MODEL[landmark]

    return df[[f'{landmark_index}_x', f'{landmark_index}_y', f'{landmark_index}_z']].values


def generate_angle_feature(df, out, landmark_a, landmark_b, landmark_c):
    out[f'{landmark_a}_to_{landmark_b}_to_{landmark_c}'] = calculate_angle(get_point(df, landmark_a),
                                                                           get_point(df, landmark_b),
                                                                           get_point(df, landmark_c))


def df_to_angles_df(data):
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
    angles_df['label'] = data['label']
    return angles_df


train, test = read_dataset(class_names)

train = df_to_angles_df(train)
test = df_to_angles_df(test)

X_train = train.drop(['label'], axis=1)
y_train = train['label']
X_test = test.drop(['label'], axis=1)
y_test = test['label']

if __name__ == "__main__":
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

    save_dir = 'outputs'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    dump(automl, f"{save_dir}/automl_model.pkl")
