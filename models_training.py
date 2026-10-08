# train.py
# Facility Accessibility Classification (Midterm Group 28)
#
# What this script does:
#   1. Loads the CSV dataset
#   2. Checks the data (missing values, duplicates, wrong labels)
#   3. Turns Yes/No into 1/0 so the models can read it
#   4. Splits the data into a training part and a testing part
#   5. Trains two models: Decision Tree and Random Forest
#   6. Tests both models and prints their scores
#   7. Picks the better model
#   8. Saves everything into model_bundle.joblib for the Streamlit app
#
# How to run it (open the terminal inside this folder):
#   python train.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# root folder address (where the script is saved at)
folder_path = os.path.dirname(os.path.abspath(__file__))

# folder location for ze files. Either for reading or saving the files below
csv_file_path = os.path.join(folder_path, "facilities.csv")
bundle_file_path = os.path.join(folder_path, "model_bundle.joblib")
results_file_path = os.path.join(folder_path, "comparison_results.csv")

# for a "stable" randomization
random_seed = 42

# testing range, rest for training
test_size = 0.2

# reference column for the model
target_column = "accessibility_label"

# inputs that the model will use
yes_no_columns = ["ramp", "elevator", "accessible_toilet", "handrails", "accessible_parking"]
number_columns = ["doorway_width_cm"]
feature_columns = yes_no_columns + number_columns

# model output
label_names = ["Accessible", "Partially Accessible", "Not Accessible"]

# yes or no "normalization"
yes_no_map = {"Yes": 1, "No": 0}

# loads the data set
data = pd.read_csv(csv_file_path)
print("Dataset Loaded - containing " + str(len(data)) + " rows and " + str(data.shape[1]) + " columns")
print("")

# data checking
# checks for missing values
print("Missing values in each column:")
print(data.isna().sum())
print("")

# checks of duplicate rows
print("Number of duplicate rows: " + str(data.duplicated().sum()))
print("")

# invalid label check
for one_label in data[target_column]:
    if one_label not in label_names:
        print("ERROR: found an invalid label: " + str(one_label))
        raise SystemExit

# Yes or No column check
for column_name in yes_no_columns:
    for one_value in data[column_name]:
        if one_value not in yes_no_map:
            print("ERROR: invalid value '" + str(one_value) + "' in column " + column_name)
            raise SystemExit

# doorway width check
for one_width in data["doorway_width_cm"]:
    if one_width <= 0:
        print("ERROR: doorway width must be above 0, found: " + str(one_width))
        raise SystemExit

# counts how many accessbile, not accessible, and partially accessible on the accessiblity_label column
print("Accessibility Lable Count:") 
print(data[target_column].value_counts())
print("")

# X is the table of inputs 
# y is the table of outputs
X = data[feature_columns].copy()
y = data[target_column]

# turn Yes into 1 and No into 0, one column at a time
for column_name in yes_no_columns:
    X[column_name] = X[column_name].map(yes_no_map)

print("Yes/No Values -> 1/0's Sucess!")
print("Dataset first 5 rows changes sample:") # TODO change
print(X.head())
print("")

# splitting time
# training set = model will learn from this
# test set = model will use this as sample of their learning
# stratify=y keeps the label balance similar in both parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    stratify=y,
    random_state=random_seed
)

print("Dataset Splitting Results:")
print("Training rows: " + str(len(X_train)))
print("Testing rows: " + str(len(X_test)))
print("Model Training is on-going!")
print("")
print("Model Training is done!")

# summon the models
decision_tree_model = DecisionTreeClassifier(random_state=random_seed)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=random_seed)

# dictionary for looping the models
all_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

# for cross-validation: split the data into 5 parts and test 5 times
cross_validation_splitter = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=random_seed
)

all_metrics = {}    # i will hold the scores of each model
all_confusion = {}  # i will hold the confusion matrix of each model

for model_name in all_models:

    current_model = all_models[model_name]

    # train the model using the training data
    current_model.fit(X_train, y_train)

    # let the model guess the labels of the testing data
    predictions = current_model.predict(X_test)

    # calculate the evaluation metric scores
    # "macro" means each of the 3 labels counts equally
    accuracy_value = accuracy_score(y_test, predictions)                                        # percentage of right guesses
    precision_value = precision_score(y_test, predictions, average="macro", zero_division=0)    # when it says "Accessible," is it really?
    recall_value = recall_score(y_test, predictions, average="macro", zero_division=0)          # out of all accessible facilities, how many did it catch
    f1_value = f1_score(y_test, predictions, average="macro", zero_division=0)                  # balances out the precision and recall

    # cross-validation score (average F1 over the 5 tests)
    cross_scores = cross_val_score(
        current_model, X, y, cv=cross_validation_splitter, scoring="f1_macro"
    )
    cross_average = cross_scores.mean()

    # save the scores 
    all_metrics[model_name] = {
        "accuracy": float(accuracy_value),
        "precision": float(precision_value),
        "recall": float(recall_value),
        "f1": float(f1_value),
        "cv_f1": float(cross_average)
    }

    # confusion matrix
    # rows = actual label 
    # columns = predicted label
    matrix = confusion_matrix(y_test, predictions, labels=label_names)
    all_confusion[model_name] = matrix

    # print the results of this model
    print("Results of " + model_name)
    print(classification_report(y_test, predictions, labels=label_names, zero_division=0))

    print("Confusion matrix:")
    print(pd.DataFrame(matrix, index=label_names, columns=label_names))
    print("")

    print("5-fold cross-validation F1: " + str(round(cross_average, 3)))
    print("")

# model comparison
comparison_table = pd.DataFrame(all_metrics).T  

print("MODEL COMPARISON TABLE:")
print(comparison_table.round(3))
print("")

decision_tree_cv = all_metrics["Decision Tree"]["cv_f1"]
random_forest_cv = all_metrics["Random Forest"]["cv_f1"]

# rule: the higher cross-validation F1 wins
# if the two are almost equal (difference of 0.01 or less), 
# pick the Decision Tree because it is easier to explain
if random_forest_cv - decision_tree_cv > 0.01:
    best_model_name = "Random Forest"
else:
    best_model_name = "Decision Tree"

print("Best overall model: " + best_model_name)
print("")

# which inputs matters the most
all_importances = {}

for model_name in all_models:

    current_model = all_models[model_name]
    importance_values = current_model.feature_importances_ # i get the importance values of the inputs

    # match each importance number with its column name
    importance_for_this_model = {}
    index = 0
    for column_name in feature_columns:
        importance_for_this_model[column_name] = float(importance_values[index])
        index = index + 1

    all_importances[model_name] = importance_for_this_model

    print("Feature importance - " + model_name)
    print(pd.Series(importance_for_this_model).sort_values(ascending=False).round(3))
    print("")

# saving model for streamlit app
# the confusion matrices are numpy arrays, so turn them into normal lists
confusion_as_lists = {}
for model_name in all_confusion:
    confusion_as_lists[model_name] = all_confusion[model_name].tolist()

# everything goes into one dictionary, then one joblib file
bundle = {
    "models": all_models,                    # the trained Decision Tree and Random Forest
    "metrics": all_metrics,                  # accuracy, precision, recall, f1, cv_f1
    "confusion_matrices": confusion_as_lists,
    "labels": label_names,
    "features": feature_columns,             # the app must give inputs in this order
    "yes_no_map": yes_no_map,                # the app must turn Yes/No into numbers the same way
    "best_model": best_model_name,
    "feature_importances": all_importances
}

joblib.dump(bundle, bundle_file_path)

print("Saved model_bundle.joblib")
print("Done!")