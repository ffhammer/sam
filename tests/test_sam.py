from copy import deepcopy

import matplotlib.pyplot as plt
import pytest

from sam.data_formats import load_datapoints
from sam.stress_addition_model import STANDARD_SAM_SETTING, SAMPrediction

SETTINGS = STANDARD_SAM_SETTING


@pytest.mark.parametrize("datapoint", load_datapoints())
def test_dose_response_fit_and_plot(datapoint):
    path, data, name, val = datapoint

    frozen_main_series = deepcopy(data.main_series)
    frozen_val_series = deepcopy(val)

    res = SAMPrediction.generate(
        data.main_series,
        val,
        data.meta,
        settings=SETTINGS,
    )

    assert frozen_main_series == res.control.inputs, "Mutated Data"
    assert frozen_val_series == res.co_stressor.inputs, "Mutated Data"

    res.plot(with_lcs=True)
    plt.close()
