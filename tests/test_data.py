import os
from pathlib import Path
import glob
import pytest
from sam.dose_reponse_fit import dose_response_fit, ModelPredictions, StandardSettings
from sam.plotting import plot_fit_prediction
from sam.data_formats import ExperimentData, read_data
from sam import REPO_PATH

@pytest.fixture(scope="module", autouse=True)
def change_to_repo_dir():
    # Change to the repository root directory
    os.chdir(REPO_PATH)


@pytest.mark.parametrize("path", glob.glob("data/*.xlsx"))
def test_loading_data(path):
    # Read data
    data: ExperimentData = read_data(path)

