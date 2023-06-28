"""
Module to ingest and merge multiple data
Author: Moises Gonzalez
Date: 28/Jun/2023
"""

import pandas as pd
import logging
import json
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
today = datetime.today().strftime("%Y%M%d")



#############Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe() -> None:
    """
    Function to check for datasets, compile them together
    and write to an output file
    """

    merged_data = pd.DataFrame()
    ingested_files = list()

    output_files_path = Path(output_folder_path)
    output_files_path.mkdir(parents=True, exist_ok=True)

    data_files_path = Path(input_folder_path).glob("*.csv")
    files = [x for x in data_files_path if x.is_file()]
    logger.info(f"Files in data path: {files}")

    for file in files:
        temp = pd.read_csv(file)
        merged_data = pd.concat([merged_data, temp], axis=0)
        ingested_files.append(str(file))

    merged_data.drop_duplicates(inplace=True)
    merged_data_csv = output_files_path / "merged_data.csv"
    merged_data.to_csv(merged_data_csv, index=False)
    ingested_files_txt = output_files_path / "ingested_files.txt"

    with open(ingested_files_txt, "w") as txt:
        txt.write(str(ingested_files))


if __name__ == '__main__':
    merge_multiple_dataframe()
