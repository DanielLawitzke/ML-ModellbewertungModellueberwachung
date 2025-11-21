import pandas as pd
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


##################Function to get model predictions
def model_predictions():
    # read the deployed model and a test dataset, calculate predictions

    # Load deployed model
    model_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_file = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(test_file)

    # Make predictions
    X_test = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    predictions = model.predict(X_test)

    return (
        predictions.tolist()
    )  # return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here

    # Load ingested data
    data_file = os.path.join(dataset_csv_path, "finaldata.csv")
    df = pd.read_csv(data_file)

    # Calculate statistics for numeric columns
    numeric_columns = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]

    statistics = []
    for column in numeric_columns:
        statistics.append(float(df[column].mean()))
        statistics.append(float(df[column].median()))
        statistics.append(float(df[column].std()))

    return statistics  # return value should be a list containing all summary statistics


##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py

    # Time ingestion.py
    start = timeit.default_timer()
    import ingestion

    ingestion.merge_multiple_dataframe()
    ingestion_time = timeit.default_timer() - start

    # Time training.py
    start = timeit.default_timer()
    import training

    training.train_model()
    training_time = timeit.default_timer() - start

    return [
        ingestion_time,
        training_time,
    ]  # return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list():
    # get a list of

    result = subprocess.run(
        ["pip", "list", "--outdated"], capture_output=True, text=True
    )

    return result.stdout


if __name__ == "__main__":
    print("Model Predictions:", model_predictions())
    print("Summary Statistics:", dataframe_summary())
    print("Execution Times:", execution_time())
    print("Outdated Packages:\n", outdated_packages_list())
