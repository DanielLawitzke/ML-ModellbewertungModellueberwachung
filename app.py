from flask import Flask, jsonify, request
import pandas as pd
import pickle
import os
import json
import diagnostics
import scoring


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    # call the prediction function you created in Step 3

    # Get filepath from request
    filepath = request.json.get("filepath")

    # Load data
    df = pd.read_csv(filepath)

    # Load deployed model
    model_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    X = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    predictions = model.predict(X)

    return jsonify(predictions.tolist())  # add return value for prediction outputs


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def score():
    # check the score of the deployed model

    # Get F1 score from deployed model
    f1_score = scoring.score_model()
    return jsonify(f1_score)  # add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    # check means, medians, and modes for each column

    # Get summary statistics
    statistics = diagnostics.dataframe_summary()
    return jsonify(statistics)  # return a list of all calculated summary statistics


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics_endpoint():
    # check timing and percent NA values

    # Get execution times
    timings = diagnostics.execution_time()

    # Calculate missing data percentages
    data_file = os.path.join(dataset_csv_path, "finaldata.csv")
    df = pd.read_csv(data_file)
    missing_percentages = (df.isna().sum() / len(df) * 100).tolist()

    # Get outdated packages
    outdated = diagnostics.outdated_packages_list()

    return jsonify(
        {
            "execution_times": timings,
            "missing_data_percentages": missing_percentages,
            "outdated_packages": outdated,
        }
    )  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
