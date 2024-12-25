import pytest
from sam.dose_reponse_fit import dose_response_fit, ModelPredictions, DRF_Settings
from sam.plotting import plot_fit_prediction
from sam.data_formats import ExperimentData, read_data, load_files
from copy import deepcopy
from itertools import product
import matplotlib.pyplot as plt


@pytest.mark.parametrize("file", product(load_files(), ("lmcurve", "scipy")))
def test_dose_response_fit_and_plot(file):
    (path, data), curve_fit_lib = file

    data: ExperimentData = read_data(path)

    frozen_inputs = deepcopy(data.main_series)

    # Perform the model fitting
    res: ModelPredictions = dose_response_fit(
        data.main_series,
        DRF_Settings(max_survival=data.meta.max_survival, curve_fit_lib=curve_fit_lib),
    )

    assert res.inputs == frozen_inputs, "Mutated Data"

    # Create the plot
    title = f"{data.meta.main_stressor} - {data.meta.organism}"
    fig = plot_fit_prediction(model=res, title=title)

    # Save the plot
    plt.close()
