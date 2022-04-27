import pandas as pd
import sklearn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from pathlib import Path
from joblib import dump, load
import glob
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from tqdm import tqdm
import beepy

# Create dataset
from utils import FakeTrial

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

def compute_acc(config):
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

    clf_names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        # "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        "XGBoost"
    ]

    # XGBoost: Set parameters found by optuna
    trial = FakeTrial({'booster': 'dart', 'lambda': 8.082457988671528e-06, 'alpha': 3.9239035875153386e-07,
                       'subsample': 0.25131589916254415, 'colsample_bytree': 0.5467022833532039, 'max_depth': 3,
                       'min_child_weight': 7, 'eta': 0.15197226233426275, 'gamma': 2.6797080962357936e-08,
                       'grow_policy': 'lossguide', 'sample_type': 'uniform', 'normalize_type': 'tree',
                       'rate_drop': 0.0035888275126407594, 'skip_drop': 0.05678533615073127})
    xgb_params = {
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": len(class_names),
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "tree_method": 'gpu_hist',
        'gpu_id': 0,
        'num_boost_round': 100,
        # 'early_stopping_rounds': 25,
        'seed': 108
    }
    if xgb_params["booster"] == "gbtree" or xgb_params["booster"] == "dart":
        xgb_params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        # minimum child weight, larger the term more conservative the tree.
        xgb_params["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        xgb_params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        xgb_params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        xgb_params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if xgb_params["booster"] == "dart":
        xgb_params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        xgb_params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        xgb_params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        xgb_params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Decision Tree: Set parameters found by optuna
    dt_params = {'criterion': 'gini', 'splitter': 'best', 'max_depth': 11, 'min_samples_split': 0.0843463691913969,
                 'min_samples_leaf': 0.017900502267895104, 'min_weight_fraction_leaf': 0.010145940025576516,
                 'max_features': 'sqrt', 'class_weight': None}

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma='scale', C=1, class_weight='balanced'),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(**dt_params),
        RandomForestClassifier(max_depth=12, n_estimators=400, max_features=12),
        MLPClassifier(alpha=1e-2, hidden_layer_sizes=(150,), max_iter=4000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        XGBClassifier(**xgb_params)
    ]

    scores = []
    for name, clf in zip(clf_names, classifiers):
        clf.fit(X_train.to_numpy(), y_train.to_numpy())
        acc = clf.score(X_test.to_numpy(), y_test.to_numpy())
        scores.append(acc)
        print(f"{name}: Accuracy = {acc * 100:.5}%")
        print(sklearn.metrics.confusion_matrix(y_test.to_numpy(), clf.predict(X_test.to_numpy())))

    scores_df = pd.DataFrame(data=[scores], columns=clf_names)
    # scores_df.to_csv(f"{output_dir}/direct_clf_scores.csv")
    # print(scores_df)
    #
    # dump(classifiers[clf_names.index("Nearest Neighbors")], f"{output_dir}/model.pkl")

    print(f"Max Score: {np.array(scores).max()}")
    print(clf_names[np.array(scores).argmax()])

    return np.array(scores).max()

output_dir = "outputs"
Path(output_dir).mkdir(exist_ok=True, parents=True)

n_trials = 20
configs = [{'A': True, 'P': False, 'C': False},
           {'A': True, 'P': True, 'C': False},
           {'A': True, 'P': False, 'C': True},
           {'A': True, 'P': True, 'C': True},
           {'A': False, 'P': True, 'C': False},
           {'A': False, 'P': True, 'C': True},
           ]

name_str = lambda x: "+".join([k for k, v in x.items() if v])
configs_str = list(map(name_str, configs))
results = pd.DataFrame(columns=configs_str)

with tqdm(total=len(configs) * n_trials) as pbar:
    for config_str, config in tqdm(zip(configs_str, configs)):
        print(f"Running config: {config_str}")
        for trial_num in tqdm(range(n_trials)):
            print(f"Trial no: {trial_num}")
            acc = compute_acc(config)
            print(f"Accuracy: {acc}")
            results.loc[trial_num, config_str] = acc
            pbar.update(1)

print(results)
results.to_csv(f"{output_dir}/dataset_selection_results2.csv")

beepy.beep("ready")