"""
Module to implement the automation of
the ML model scoring, monitoring and re-deployment
Author: Moises Gonzalez
Date: 02/Jul/2023
"""

import logging
import json
from pathlib import Path

import ingestion
import training
import scoring
import deployment
import reporting
import diagnostics
from apicalls import call_api_endpoint

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug.log"),
    ],
)
logger = logging.getLogger(__name__)


with open("config.json", "r") as f:
    config = json.load(f)

production_deployment_path = Path(config["prod_deployment_path"])
input_folder_path = Path(config["input_folder_path"])
output_folder_path = Path(config["output_folder_path"])
model_path = Path(config["output_model_path"])

ingested_files_txt = production_deployment_path / "ingestedfiles.txt"

with open(ingested_files_txt, "r") as f:
    ingested_files = f.read()

data_files_path = input_folder_path.glob("*.csv")
files = [x.stem for x in data_files_path if x.is_file()]
logger.info(f"Checking for new data...")
logger.info(f"Files in input data path: {files}")
logger.info(f'Previously ingested data files -> {ingested_files.strip(",")}')

non_ingested_status = [x not in ingested_files for x in files]
files_non_ingested = {files[i]: non_ingested_status[i] for i in range(len(files)) if non_ingested_status[i]} # noqa

logger.info(f"Files non ingested yet -> {files_non_ingested}")

if len(files_non_ingested) > 0:
    ingestion.merge_multiple_dataframe()

    latest_score_txt = production_deployment_path / "latestscore.txt"
    with open(latest_score_txt, "r") as f:
        latest_score = float(f.read())

    logger.info(f"Latest score was -> {latest_score}")

    new_F1_score = scoring.score_model(save_file=False)
    logger.info(f"New score is -> {new_F1_score}")

    if new_F1_score >= latest_score:
        new_F1_score = scoring.score_model(save_file=True)
        logger.info("No Model Drift!")
    else:
        logger.info("Model Drift occurred!")
        logger.info("Re-training model")
        training.train_model()
        logger.info("Re-deploying model")
        deployment.store_model_into_pickle()
        logger.info("Re-generating confusion matrix")
        reporting.score_model()
        logger.info(f"Calling diagnostics")
        diagnostics.model_predictions()
        diagnostics.dataframe_summary()
        diagnostics.execution_time()
        diagnostics.outdated_packages_list()
        diagnostics.missing_data()
        logger.info(f"Calling API endpoints")
        call_api_endpoint(url="http://127.0.0.1:8000")
        logger.info(f"Re-deployment complete!")
