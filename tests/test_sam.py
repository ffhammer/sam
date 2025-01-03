from sam.stress_addition_model import generate_sam_prediction, STANDARD_SAM_SETTING
from sam.data_formats import load_datapoints
import pytest
from copy import deepcopy

SETTINGS = STANDARD_SAM_SETTING


@pytest.mark.parametrize("datapoint", load_datapoints())
def test_dose_response_fit_and_plot(datapoint):
    path, data, name, val = datapoint

    frozen_main_series = deepcopy(data.main_series)
    frozen_val_series = deepcopy(val)

    res = generate_sam_prediction(
        data.main_series,
        val,
        data.meta,
        settings=SETTINGS,
    )

    assert frozen_main_series == res.control.inputs, "Mutated Data"
    assert frozen_val_series == res.co_stressor.inputs, "Mutated Data"
