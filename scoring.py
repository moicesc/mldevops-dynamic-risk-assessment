"""
Module to perform the model scoring
Author: Moises Gonzalez
Date: 29/Jun/2023
"""


import pandas as pd
import pickle
import json
import logging

from sklearn import metrics
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


logger.info("Load config.json and get path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])
test_data_path = Path(config['test_data_path'])
model_path = Path(config['output_model_path'])

model_file = model_path / "trainedmodel.pkl"
test_data_file = test_data_path / "testdata.csv"


def score_model(model: Path = model_file) -> float:
    """
    Function to calculate the F1 score of a defined model

    Args:
        model: Path to the model to get the score

    Returns:
        F1 score calculated from the given model
    """

    data = dataset_csv_path / "finaldata.csv"
    df = pd.read_csv(data)
    logger.info(f"Data loaded -> {data}")

    df = df.drop("corporation", axis=1)
    y = df.exited.values
    X = df.drop("exited", axis=1)

    with open(model_file, "rb") as f:
        model_lr = pickle.load(f)
    logger.info(f"Model loaded -> {model_file}")

    preds = model_lr.predict(X)
    logger.info("Predictions complete")

    f1_score = metrics.f1_score(y_true=y, y_pred=preds)
    logger.info(f"F1 score -> {f1_score}")
    f1_score_txt = model_path / "latestscore.txt"
    with open(f1_score_txt, "w") as txt:
        txt.write(str(f1_score))
    logger.info(f"F1 score file -> {f1_score_txt}")

    return f1_score


if __name__ == "__main__":
    score_model()
