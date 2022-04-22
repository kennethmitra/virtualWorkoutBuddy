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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Create dataset
class_names = ["pushup_up", "pushup_down", "situp_up", "situp_down", "squat_up", "squat_down"]
data_files = ["data/pushup_up.csv", "data/pushup_down.csv", "data/situp_up.csv", "data/situp_down.csv", "data/squat_up.csv", "data/squat_down.csv"]

per_class_dfs = []
for i, class_name in enumerate(class_names):
    df = pd.read_csv(data_files[i], index_col=0)
    df['label'] = i
    df = df.iloc[::1, :]
    per_class_dfs.append(df)
    print(f"{class_name} - {len(df)}")

min_len = min(map(len, per_class_dfs))
per_class_dfs = [df.iloc[:min_len, :] for df in per_class_dfs]

for df in per_class_dfs:
    print(f"{len(df)}")

data = pd.concat(per_class_dfs, ignore_index=True)
dataX = data.loc[:, data.columns != 'label']
dataY = data['label']

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.25, stratify=dataY)

clf_names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    #"Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1e-2, max_iter=4000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

scores = []
for name, clf in zip(clf_names, classifiers):
    clf.fit(X_train.to_numpy(), y_train.to_numpy())
    acc = clf.score(X_test.to_numpy(), y_test.to_numpy())
    scores.append(acc)
    print(f"{name}: Accuracy = {acc*100:.5}%")

output_dir = "outputs"
Path(output_dir).mkdir(exist_ok=True, parents=True)
scores_df = pd.DataFrame(data=[scores], columns=clf_names)
scores_df.to_csv(f"{output_dir}/direct_clf_scores.csv")
print(scores_df)

dump(classifiers[clf_names.index("Nearest Neighbors")], f"{output_dir}/model.pkl")