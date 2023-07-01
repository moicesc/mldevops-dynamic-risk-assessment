"""
Module to generate a Flask API for model reporting
Author: Moises Gonzalez
Date:01/Jul/2023
"""

from flask import Flask, session, jsonify, request
from pathlib import Path
from diagnostics import model_predictions, dataframe_summary,\
    execution_time, missing_data
from scoring import score_model
import json


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path'])

prediction_model = None


@app.route("/")
def welcome():
    return f"Welcome to the reports! \n " \
           f"/prediction, /scoring, /summarystats, or /diagnostics"


@app.route("/prediction", methods=['POST', 'OPTIONS', 'GET'])
def predict():
    """
    Prediction endpoint
    """

    data = request.args.get("data")
    predictions = model_predictions(data)

    return f"predictions -> {predictions}"


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """
    Score endpoint
    """

    model_score = score_model()

    return f"F1 score is -> {model_score}"


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    Summary statistics endpoint
    check means, medians, and modes for each column
    """

    data = request.args.get("data")
    statisticss = dataframe_summary(data)

    return f"Means, medians and standard deviations -> {statisticss}"


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diag():
    """
    Diagnostics endpoint
    chekc timing and percent of NA values
    """

    data = request.args.get("data")
    missing = missing_data(data)
    timing = execution_time()

    diagnostics = {"missing_data": missing,
                   "execution_time": timing}

    return diagnostics


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
