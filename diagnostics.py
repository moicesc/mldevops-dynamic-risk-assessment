"""
Module to perform the diagnostics of the model
Author: Moises Gonzalez
Date: 30/jun/2023
"""
import pandas as pd
import numpy as np
import timeit
import pickle
import json
import logging
import subprocess

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Load config.json and get environment variables")
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])
test_data_path = Path(config['test_data_path'])
production_deployment_path = Path(config['prod_deployment_path'])


def model_predictions(test_dataset=None) -> list:
    """
    Function to perform predictions on a given test dataset

    Args:
        test_dataset: a test dataset to perform predictions to

    Returns:
        A list with all the predictions
    """

    logger.info("[DIAG] Running model predictions")
    if test_dataset is None:
        test_dataset_file = test_data_path / "testdata.csv"
        logger.info(f"Using default test dataset -> {test_dataset_file}")
        test_dataset = pd.read_csv(test_dataset_file)

    model_file = production_deployment_path / "trainedmodel.pkl"
    with open(model_file, "rb") as m:
        model = pickle.load(m)
    logger.info(f"Model loaded -> {model_file}")

    test_dataset = test_dataset.drop("corporation", axis=1)
    X = test_dataset.drop("exited", axis=1)

    predictions = model.predict(X)
    logger.info("Predictions done")

    return predictions


def dataframe_summary() -> list:
    """
    Function to calculate data statistics
    Returns:
        A list containing all summary statistics
    """

    logger.info("[DIAG] Getting data statistics")
    dataset_file = dataset_csv_path / "finaldata.csv"
    df = pd.read_csv(dataset_file)

    num_col_index = np.where(df.dtypes != object)[0]
    num_col = df.columns[num_col_index].tolist()

    means = df[num_col].mean(axis=0).tolist()
    medians = df[num_col].median(axis=0).tolist()
    std_devs = df[num_col].std(axis=0).tolist()

    logger.info(f"means -> {means}")
    logger.info(f"medians -> {medians}")
    logger.info(f"std_devs -> {std_devs}")

    stats = means + medians + std_devs

    logger.info(f"Statistics calculated {stats}")

    return stats


def missing_data(df=None) -> list:
    """
    Function to check for missing data

    Args:
        df: Dataframe to check for missing data
    Returns:
        A list with the percent of NA values in a data column
    """

    logger.info("[DIAG] Getting missing data percentage")
    if df is None:
        df_file = dataset_csv_path / "finaldata.csv"
        logger.info(f"Using default dataset -> {df_file}")
        df = pd.read_csv(df_file)

    missing = df.isna().sum(axis=0) / len(df) * 100

    logger.info(f"Missing values -> {missing.values}")

    return missing.values


def execution_time() -> list:
    """
    Calculates the execution time for the data ingestion
    and model training

    Returns:
        A list containing the execution time for each step
    """

    logger.info("[DIAG] Getting execution time")
    start_t = timeit.default_timer()
    subprocess.run(["python", "ingestion.py"])
    end_t = timeit.default_timer()

    duration_ingestion = end_t - start_t

    start_t = timeit.default_timer()
    subprocess.run(["python", "training.py"])
    end_t = timeit.default_timer()

    duration_training = end_t - start_t

    logger.info(f"Data ingestion executed in -> {duration_ingestion}s")
    logger.info(f"Model training executed in -> {duration_training}s")

    return [duration_ingestion, duration_training]


def outdated_packages_list() -> str:
    """
    Function to get outdated dependencies
    Returns:
        A string with the outdated packages
    """

    logger.info("[DIAG] Getting outdated packages")
    outdated = subprocess.run(
        ["python", "-m",  "pip", "list", "--outdated"],
        capture_output=True).stdout

    logger.info(
        f"Outdated packages retrieved -> \n {outdated.decode('utf-8')}\n")

    return outdated.decode('utf-8')


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
    missing_data()
