import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score
import json


#################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])


#################Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    # Load trained model
    model_file = os.path.join(model_path, "trainedmodel.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_file = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(test_file)

    # Prepare features and target
    X_test = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y_test = df["exited"]

    # Make predictions and calculate F1 score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    # Save F1 score
    score_file = os.path.join(model_path, "latestscore.txt")
    with open(score_file, "w") as f:
        f.write(str(f1))

    return f1


if __name__ == "__main__":
    score = score_model()
    print(f"F1 Score: {score}")
