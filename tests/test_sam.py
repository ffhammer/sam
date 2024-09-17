from sam.stress_addition_model import (
    sam_prediction,
    OLD_STANDARD
)
from sam.data_formats import load_datapoints
from tqdm import tqdm
import pytest

SETTINGS = OLD_STANDARD


@pytest.mark.parametrize("datapoint", load_datapoints())
def test_dose_response_fit_and_plot(datapoint):
    path, data, name, val = datapoint

    main_fit, stress_fit, sam_sur, sam_stress, additional_stress = sam_prediction(
        data.main_series,
        val,
        data.meta,
        settings=SETTINGS,
    )
