import glob
import pytest
from sam.data_formats import ExperimentData, read_data, load_datapoints
from tqdm import tqdm


@pytest.mark.parametrize("path", glob.glob("data/*/*.xlsx"))
def test_loading_data(path):
    # Read data
    data: ExperimentData = read_data(path)
    data.hormesis_index  # chexk data

    assert (
        data.meta.max_survival > 0
    ), f"Max survival has value {data.meta.max_survival}"
    assert all(data.main_series.survival_rate >= 0), f"Survival Rate must be >= 0"
    assert all(
        data.main_series.survival_rate <= data.meta.max_survival
    ), f"Survival Rate must be <= max_survival"

    for ser in data.additional_stress.values():
        assert all(ser.survival_rate >= 0), f"Survival Rate must be >= 0"
        assert all(
            ser.survival_rate <= data.meta.max_survival
        ), f"Survival Rate must be <= max_survival"


def test_loading_datapoints():
    for _ in tqdm(load_datapoints()):
        pass
