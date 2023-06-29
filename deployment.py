"""
Module to copy model to deployment directory
Author: Moises Gonzalez
Date: 30/Jun/2023
"""
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logging.info("Load config.json and correct path variable")
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])
model_path = Path(config['output_model_path'])
prod_deployment_path = Path(config['prod_deployment_path'])


def store_model_into_pickle():
    """
    copy the latest pickle file, the latestscore.txt value,
    and the ingestfiles.txt file into the deployment directory
    """

    files_to_copy = [model_path / "trainedmodel.pkl",
                     model_path / "latestscore.txt",
                     dataset_csv_path / "ingestedfiles.txt"]

    file_names = ["trainedmodel.pkl", "latestscore.txt", "ingestedfiles.txt"]

    prod_deployment_path.mkdir(parents=True, exist_ok=True)

    for source_file, dst_file in zip(files_to_copy, file_names):
        shutil.copy(source_file, prod_deployment_path / dst_file)
        logger.info(f"File copied to -> {prod_deployment_path / dst_file}")


if __name__ == "__main__":
    store_model_into_pickle()
