# Virtual Workout Buddy

## Quick Start
### For Best Results
Ensure your webcam can see your entire body as you perform squat, situp, and pushup exercises. For best results, when performing situps, angle yourself so the webcam program can detect your feet.
Virtual Workout Buddy may produce sporadic results if your entire body is not in the frame. Make sure you are ready to begin the workout when launching the program.
### Launching Virtual Workout Buddy
To install the required python libraries on your system, run `pip install -r requirements.txt`

To launch the application using the pretrained model, run `main_gui.py`
## File Structure

### Training / Validation Datasets
```
data/                       # Files in this directory are created by running data_collection.py
├── pushup_down.csv         # Training data with label pushup_down
├── pushup_up.csv           # Training data with label pushup_up
├── situp_down.csv          # Training data with label situp_down
├── situp_up.csv            # Training data with label situp_up
├── squat_down.csv          # Training data with label squat_down
├── squat_up.csv            # Training data with label squat_up
├── test_pushup_down.csv    # Validation data with label pushup_down
├── test_pushup_up.csv      # Validation data with label pushup_up
├── test_situp_down.csv     # Validation data with label situp_down
├── test_situp_up.csv       # Validation data with label situp_up
├── test_squat_down.csv     # Validation data with label squat_down
└── test_squat_up.csv       # Validation data with label squat_up
```

### Outputs directory
```
outputs/
├── conf_mats/                       # Directory containing confusion matricies (pngs) for each model type
├── xgboost_optuna_visualizations/   # Directory containing visualizations for optuna hyperparameter search (for XGBoost model)
├── clf_scores.csv                   # CSV file with accuracies of each classifier over selected dataset (created by classifier.py)
├── dataset_selection_results.csv    # CSV file with accuracy of top performing classifier over each dataset combination (created by dataset_selection.py)
├── dt_clf_out.txt                   # Log file for Decision Tree Optuna hyperparameter search (created by dt_clf.py)
├── xgboost_clf_out.txt              # Log file for XGBoost Optuna hyperparameter search (created by xgboost_clf.py)
└── model.pkl                       # Pickle file storing top performing classifier to be used in main_gui.py (created by classifier.py)
```

### Main Directory Code
```
.
├── classifier.py                       # Script used to train classifiers and report accuracies. Allows for dataset selection and outputs confusion matricies. Saves highest scoring model in outputs/model.pkl
├── config.json                         # JSON configuration file specifying number of reps of each exercise to be done as well as order of exercises (used by main_gui.py)
├── data_collection.py                  # Script used to generate training and validation datasets. Records poses while 'z' key is held down. Press 'esc' to write data to disk.
├── dataset_selection.py                # Script that runs classifiers over every dataset combination. Saves results of 20 trials to outputs/dataset_selection_results.csv
├── dt_clf.py                           # Script that runs Optuna hyperparameter search for decision tree classifier
├── main_gui.py                         # Main script that launches Virtual Workout Buddy GUI
├── requirements.txt                    # Lists python libraries required to run all scripts in this project
├── utils.py                            # Script containing utility functions used in other scripts
├── xgboost_clf.py                      # Script that runs Optuna hyperparameter search for XGBoost classifier
└── xgboost_clf_study.pkl               # Pickle file storing Optuna study object for XGBoost.
```