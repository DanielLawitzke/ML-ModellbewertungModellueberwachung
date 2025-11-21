import os
import json
import ingestion
import training
import scoring
import deployment
import reporting
import pdf_report


with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = "sourcedata"  # Hardcoded config['input_folder_path'] won't work
output_folder_path = config["output_folder_path"]
prod_deployment_path = config["prod_deployment_path"]


##################Check and read new data
# first, read ingestedfiles.txt
ingested_files_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
with open(ingested_files_path, "r") as f:
    ingested_files = set(f.read().splitlines())

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_files = set(f for f in os.listdir(input_folder_path) if f.endswith(".csv"))

new_files = source_files - ingested_files  # set substraction!

print(f"New files: {new_files}")

##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if not new_files:
    print("No new data found. Exiting.")
    exit()

print(f"New data found: {new_files}")
ingestion.merge_multiple_dataframe()

##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
deployed_score_path = os.path.join(prod_deployment_path, "latestscore.txt")
with open(deployed_score_path, "r") as f:
    deployed_score = float(f.read())

training.train_model()
new_score = scoring.score_model()

print(f"Deployed model score: {deployed_score}")
print(f"New model score: {new_score}")

##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if new_score >= deployed_score:
    print("No model drift detected. No report created. Exiting.")
    exit()

print("Model drift detected!")

##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle(None)

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
reporting.score_model()
pdf_report.generate_pdf_report()
print("PDF report created successfully!")

print("Full process completed successfully!")
