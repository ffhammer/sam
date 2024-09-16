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

@pytest.fixture(scope="module")
def setup_save_dir():
    save_dir = "control_imgs/fits"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

@pytest.mark.parametrize("path", glob.glob("data/*.xlsx"))
def test_dose_response_fit_and_plot(path, setup_save_dir):
    save_dir = setup_save_dir

    # Read data
    data: ExperimentData = read_data(path)

    # Perform the model fitting
    res: ModelPredictions = dose_response_fit(
        data.main_series, StandardSettings(survival_max=data.meta.max_survival)
    )

    # Create the plot
    title = f"{data.meta.chemical} - {data.meta.organism}"
    fig = plot_fit_prediction(model=res, title=title)

    # Save the plot
    save_path = os.path.join(save_dir, f"{os.path.split(path.replace('xlsx', 'png'))[1]}")
    fig.savefig(save_path)

    # Check if the plot was saved
    assert os.path.exists(save_path), f"Plot was not saved for {path}"
