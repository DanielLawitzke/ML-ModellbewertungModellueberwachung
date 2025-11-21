import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])


#################Function for training the model
def train_model():
    # Create model folder if not exists
    os.makedirs(model_path, exist_ok=True)

    # Load data
    data_file = os.path.join(dataset_csv_path, "finaldata.csv")
    df = pd.read_csv(data_file)

    # Prepare features and target
    X = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y = df["exited"]

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=100,
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    model.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    model_file = os.path.join(model_path, "trainedmodel.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
