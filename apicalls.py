"""
Module to call all API endpoints
Author: Moises Gonzalez
Date: 02/Jul/2023
"""
import requests
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URL = "http://127.0.0.1:8000"

with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = Path(config['test_data_path'])
output_model_path = Path(config['output_model_path'])
test_data_file = test_data_path / "testdata.csv"


def call_api_endpoint(url: str = URL):
    """
    function to call the API endpoints

    Args:
        url: url to call the endpoints to
    """

    logger.info("Calling all functions")
    response1 = requests.get(url + f"/prediction?data={str(test_data_file)}")
    response2 = requests.get(url + "/scoring")
    response3 = requests.get(url + "/summarystats")
    response4 = requests.get(url + "/diagnostics")

    responses = f"\npredictions status_code-> {response1.status_code}\n" \
                f"{response1.content.decode('utf-8')}\n" \
                f"\nscoring status_code-> {response2.status_code}\n" \
                f"{response2.content.decode('utf-8')}\n" \
                f"\nsummarystats status_code-> {response3.status_code}\n" \
                f"{response3.content.decode('utf-8')}\n" \
                f"\ndiagnostics status_code-> {response4.status_code}\n" \
                f"{response4.content.decode('utf-8')}"

    api_returns = output_model_path / "apireturns1.txt"
    with open(api_returns, "w") as txt:
        txt.write(responses)
    logger.info(f"API responses saved to -> {api_returns}")


if __name__ == "__main__":
    call_api_endpoint()
