from sam.stress_addition_model import (
    sam_prediction,
    STANDARD_SAM_SETTING
)
from sam.data_formats import load_datapoints
import pytest
from copy import deepcopy

SETTINGS = STANDARD_SAM_SETTING


@pytest.mark.parametrize("datapoint", load_datapoints())
def test_dose_response_fit_and_plot(datapoint):
    path, data, name, val = datapoint

    frozen_main_series = deepcopy(data.main_series)
    frozen_val_series = deepcopy(val)

    res  = sam_prediction(
        data.main_series,
        val,
        data.meta,
        settings=SETTINGS,
    )

    assert frozen_main_series == res.main_fit.inputs, "Mutated Data"
    assert frozen_val_series == res.stressor_fit.inputs, "Mutated Data"
    