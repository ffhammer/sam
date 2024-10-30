import os
import pytest
from sam.dose_reponse_fit import dose_response_fit, ModelPredictions, DRF_Settings
from sam.plotting import plot_fit_prediction
from sam.data_formats import ExperimentData, read_data, load_files
from sam import REPO_PATH
from copy import deepcopy
import matplotlib.pyplot as plt

@pytest.fixture(scope="module", autouse=True)
def change_to_repo_dir():
    # Change to the repository root directory
    os.chdir(REPO_PATH)


@pytest.mark.parametrize("file", load_files())
def test_dose_response_fit_and_plot(file):
    path, data = file
    
    data: ExperimentData = read_data(path)

    frozen_inputs = deepcopy(data.main_series)

    # Perform the model fitting
    res: ModelPredictions = dose_response_fit(
        data.main_series, DRF_Settings(max_survival=data.meta.max_survival)
    )


    assert res.inputs == frozen_inputs, "Mutated Data"

    # Create the plot
    title = f"{data.meta.main_stressor} - {data.meta.organism}"
    fig = plot_fit_prediction(model=res, title=title)

    # Save the plot
    plt.close()
