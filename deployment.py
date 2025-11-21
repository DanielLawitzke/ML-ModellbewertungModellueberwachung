import os
import shutil
import json


##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
model_path = os.path.join(config["output_model_path"])


####################function for deployment
def store_model_into_pickle(model):
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    # Create production deployment folder if not exists
    os.makedirs(prod_deployment_path, exist_ok=True)

    # Copy trained model
    model_source = os.path.join(model_path, "trainedmodel.pkl")
    model_dest = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    shutil.copy(model_source, model_dest)

    # Copy latest score
    score_source = os.path.join(model_path, "latestscore.txt")
    score_dest = os.path.join(prod_deployment_path, "latestscore.txt")
    shutil.copy(score_source, score_dest)

    # Copy ingested files list
    ingested_source = os.path.join(dataset_csv_path, "ingestedfiles.txt")
    ingested_dest = os.path.join(prod_deployment_path, "ingestedfiles.txt")
    shutil.copy(ingested_source, ingested_dest)


if __name__ == "__main__":
    store_model_into_pickle(None)
