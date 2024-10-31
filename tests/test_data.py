import glob
import pytest
from sam.data_formats import ExperimentData, read_data, load_datapoints
from tqdm import tqdm



@pytest.mark.parametrize("path", glob.glob("data/*.xlsx"))
def test_loading_data(path):
    # Read data
    data: ExperimentData = read_data(path)

def test_loading_datapoints():
    for _ in tqdm(load_datapoints()):
        pass