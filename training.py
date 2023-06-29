"""
Module to perform the model training
Author: Moises Gonzalez
Date: 29/Jun/2023
"""

import pandas as pd
import pickle
import json
import logging

from pathlib import Path
from sklearn.linear_model import LogisticRegression


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Load config.json and get path variables")
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])
model_path = Path(config['output_model_path'])
model_path.mkdir(parents=True, exist_ok=True)


def train_model() -> None:
    """Function to train the model"""

    data = dataset_csv_path / "finaldata.csv"
    df = pd.read_csv(data)
    logger.info(f"Data loaded -> {data}")

    df = df.drop("corporation", axis=1)
    y = df.exited.values
    X = df.drop("exited", axis=1)

    model_lr = LogisticRegression(
        C=1.0, class_weight=None,
        dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None,
        max_iter=100, multi_class='auto',
        n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001,
        verbose=0, warm_start=False)

    logger.info("Fitting model")
    model_lr.fit(X, y)
    model_pkl = model_path / "trainedmodel.pkl"
    logger.info(f"Model Trained, will be stored in -> {model_pkl}")
    pickle.dump(model_lr, open(model_pkl, "wb"))


if __name__ == "__main__":
    train_model()
