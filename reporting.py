import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os


###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
model_path = os.path.join(config["output_model_path"])


##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace

    # Load deployed model
    model_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_file = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(test_file)

    # Prepare features and target
    X_test = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y_test = df["exited"]

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save plot
    output_file = os.path.join(model_path, "confusionmatrix.png")
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    score_model()
