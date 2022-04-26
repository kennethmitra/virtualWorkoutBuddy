from pathlib import Path

import beepy
import numpy as np
import optuna
import pandas as pd

from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    def objective(trial):
        config_list = [
                {'A': True, 'P': False, 'C': False},
               {'A': True, 'P': True, 'C': False},
               {'A': True, 'P': False, 'C': True},
               {'A': True, 'P': True, 'C': True},
               {'A': False, 'P': True, 'C': False},
               {'A': False, 'P': True, 'C': True},
        ]
        config_idx = trial.suggest_categorical('config_idx', list(range(len(config_list))))
        config = config_list[config_idx]

        train, test = read_dataset(class_names)

        train_angles = df_to_angles_df(train)
        test_angles = df_to_angles_df(test)

        X_train_angles = train_angles.drop(['label'], axis=1)
        y_train_angles = train_angles['label']
        X_test_angles = test_angles.drop(['label'], axis=1)
        y_test_angles = test_angles['label']

        X_train_points = train.drop(['label'], axis=1)
        y_train_points = train['label']
        X_test_points = test.drop(['label'], axis=1)
        y_test_points = test['label']

        train_ds = []
        test_ds = []
        if config['A']:
            train_ds.append(X_train_angles)
            test_ds.append(X_test_angles)
        if config['P']:
            train_ds.append(X_train_points)
            test_ds.append(X_test_points)

        X_train = pd.concat(train_ds, axis=1)
        y_train = y_train_angles
        X_test = pd.concat(test_ds, axis=1)
        y_test = y_test_angles

        param_distributions = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'splitter': trial.suggest_categorical('splitter', ['random', 'best']),
            'max_depth': trial.suggest_int('max_depth', 1, 15),
            'min_samples_split': trial.suggest_float('min_samples_split', 0, 0.5),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0, 0.5),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
        clf = DecisionTreeClassifier(**param_distributions)
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        return (train_acc + test_acc)/2

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
            #(f"{class_name} - {len(df)}")

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


    def compute_angle_to_ground_normal(p1, p2):
        vector1 = p1 - p2
        vector2 = np.array([0, 1, 0])
        return np.sum(vector1 * vector2, axis=1) / (
                np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2)) / np.pi * 180


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


    beepy.beep("error")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10000, n_jobs=-1)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(f"Params dict: {trial.params}")

    save_dir = 'outputs'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{save_dir}/dt_clf_out.txt", "a") as f:
        print("Best trial:", file=f)
        trial = study.best_trial

        print("  Value: {}".format(trial.value), file=f)
        print("  Params: ", file=f)
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value), file=f)

        print(f"Params dict: {trial.params}", file=f)
        print("-----------------------------------------------------------------------------", file=f)

    beepy.beep("ready")