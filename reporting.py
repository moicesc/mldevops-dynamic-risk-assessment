"""
Module to evaluate and score model
Author: Moises Gonzalez
Date: 01/Jul/2023
"""

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from diagnostics import model_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])
test_data_path = Path(config['test_data_path'])
output_model_path = Path(config['output_model_path'])


def score_model() -> None:
    """
    Function to calculate a confusion matrix using the test data
    and deployed model
    """

    data = test_data_path / "testdata.csv"
    df = pd.read_csv(data)
    logger.info(f"Data loaded -> {data}")

    df = df.drop("corporation", axis=1)
    y = df.exited.values

    predictions = model_predictions()

    confusion_matrix = metrics.confusion_matrix(predictions, y)

    ax = plt.subplot()
    sns.heatmap(confusion_matrix, annot=True, ax=ax,
                cmap="Blues", fmt='', cbar=False)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    confusion_matrix_img = output_model_path / "confusion_matrix.png"
    plt.savefig(confusion_matrix_img)
    logger.info(f"Confusion Matrix saved -> {confusion_matrix_img}")


if __name__ == '__main__':
    score_model()
