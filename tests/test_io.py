import tempfile
import os
from dataclasses import asdict

import numpy as np

from sam import *
from sam import SAMPrediction
from sam.ecx_sys import generate_ecx_sys_prediction, ECxSySPrediction


def generate_example_prediction() -> SAMPrediction:
    control_series = CauseEffectData(
        concentration=[0, 0.1, 0.5, 1.0, 5.0],
        survival_rate=[100, 98, 85, 50, 10],
    )

    stressor_series = CauseEffectData(
        concentration=[0, 0.1, 0.5, 1.0, 5.0],
        survival_rate=[100, 95, 70, 30, 5],
    )

    # Run SAM prediction
    prediction: SAMPrediction = SAMPrediction.generate(
        control_data=control_series,
        co_stressor_data=stressor_series,
        settings=STANDARD_SAM_SETTING,
        max_survival=100,
    )

    return prediction


def test_saving_and_loading_stays_same_sam():
    # Generate the example prediction
    original_prediction: SAMPrediction = generate_example_prediction()

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


def test_saving_and_loading_stays_same_ecx_sys():
    # Generate the example prediction
    data = CauseEffectData(
        concentration=[0, 0.001, 0.01, 0.0316, 0.1, 0.316, 1, 3.16],
        survival_rate=[0.82, 0.74, 0.7, 0.8, 0.72, 0.53, 0.07, 0.0],
    )

    original_prediction: ECxSySPrediction = generate_ecx_sys_prediction(
        data=data, max_survival=1, hormesis_index=3
    )

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        original_prediction.save_to_file(temp_file.name)
        temp_file_path = temp_file.name

    try:
        # Deserialize
        loaded_prediction = ECxSySPrediction.load_from_file(temp_file_path)

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


def test_saving_and_loading_stays_same_for_sam_with_real_data():
    # Generate the example prediction
    data = read_data("data/2019 Naeem-Esf, Pro, food/21_days.xlsx")

    original_prediction = SAMPrediction.generate(
        data.main_series, data.additional_stress["Prochloraz_1 + Food_1%"], data.meta
    )

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
