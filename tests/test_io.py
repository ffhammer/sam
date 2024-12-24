import tempfile
import os
from dataclasses import asdict

import numpy as np

from sam import *
from sam import SAMPrediction


def generate_example_prediction() -> SAMPrediction:
    control_series = DoseResponseSeries(
        concentration=[0, 0.1, 0.5, 1.0, 5.0],
        survival_rate=[100, 98, 85, 50, 10],
        name="Control",
    )

    stressor_series = DoseResponseSeries(
        concentration=[0, 0.1, 0.5, 1.0, 5.0],
        survival_rate=[100, 95, 70, 30, 5],
        name="Stressor",
    )

    # Run SAM prediction
    prediction: SAMPrediction = sam_prediction(
        main_series=control_series,
        stressor_series=stressor_series,
        settings=STANDARD_SAM_SETTING,
        max_survival=100,
    )

    return prediction


def test_saving_and_loading_stays_same():
    # Generate the example prediction
    original_prediction : SAMPrediction= generate_example_prediction()


    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        original_prediction.save_to_file(temp_file.name)
        temp_file_path = temp_file.name

    try:
        # Deserialize
        loaded_prediction = SAMPrediction.load_from_file(temp_file_path)

        # Convert both original and loaded predictions to dictionaries
        original_dict = asdict(original_prediction)
        loaded_dict = asdict(loaded_prediction)

        # Compare the dictionaries
        assert dict_eq_manual(
            original_dict, loaded_dict
        ), "The original and loaded predictions do not match."

    finally:
        # Ensure the temporary file is deleted after the test
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



def dict_eq_manual(a: dict, b: dict) -> bool:
    # Check if both inputs are dictionaries
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False

    # Check if both dictionaries have the same keys
    if set(a.keys()) != set(b.keys()):
        return False

    # Iterate over the keys and compare values
    for key in a:
        val_a = a[key]
        val_b = b[key]

        if isinstance(val_a, dict) or isinstance(val_b, dict):
            if not dict_eq_manual(val_a, val_b):
                return False

        elif isinstance(val_a, np.ndarray) or isinstance(val_b, np.ndarray):
            if not np.isclose(val_a, val_b).all():
                return False

        else:
            if val_a != val_b:
                return False

    return True
