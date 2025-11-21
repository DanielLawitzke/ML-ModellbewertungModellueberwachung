import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


#############Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file

    # Create output folder if not exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Find all CSV files
    csv_files = [f for f in os.listdir(input_folder_path) if f.endswith(".csv")]

    # Read all CSVs
    df_list = []
    for file in csv_files:
        file_path = os.path.join(input_folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

    # Merge and remove duplicates
    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["corporation"])

    # Save finaldata.csv
    output_file = os.path.join(output_folder_path, "finaldata.csv")
    final_df.to_csv(output_file, index=False)

    # Save list of ingested files
    ingested_files_path = os.path.join(output_folder_path, "ingestedfiles.txt")
    with open(ingested_files_path, "w") as f:
        for file in csv_files:
            f.write(file + "\n")


if __name__ == "__main__":
    merge_multiple_dataframe()
