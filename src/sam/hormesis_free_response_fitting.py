from warnings import warn
from typing import Callable, Optional

import numpy as np
from py_lbfgs import lbfgs_fit

from .data_formats import CauseEffectData
from .helpers import detect_hormesis_index, fix_wlb1


def interpolate_sub_horm(
    arr,
    subhormesis_index,
    hormesis_index,
    n_new_points=3,
    log=False,
) -> np.ndarray:
    if log:
        arr_interpolated = np.logspace(
            np.log10(arr[subhormesis_index]),
            np.log10(arr[hormesis_index]),
            2 + n_new_points,
        )

    else:
        arr_interpolated = np.linspace(
            arr[subhormesis_index], arr[hormesis_index], 2 + n_new_points
        )
    return np.concatenate(
        [
            arr[:subhormesis_index],
            arr_interpolated,
            arr[hormesis_index + 1 :],
        ]
    )


def fit_hormesis_free_response(
    data: CauseEffectData,
    max_survival: float,
    hormesis_index: Optional[int],
    interpolate: bool = True,
):
    if hormesis_index is None:
        warn("Try to detect hormesis automatically")
        hormesis_index = detect_hormesis_index(data.survival_rate)

    if (
        hormesis_index is None
        or hormesis_index < 1
        or hormesis_index >= len(data.concentration)
    ):
        raise ValueError(
            f"Hormesis index '{hormesis_index}' must be 0 < index < {len(data.concentration)} == len(data)"
        )

    subhormesis_index = hormesis_index - 1

    if max_survival < 0:
        raise ValueError("Max Survival must be >= 0")

    concentrations = data.concentration
    survival_rate = data.survival_rate / max_survival

    if interpolate:
        n_new = 3
        if subhormesis_index == 0:
            concentrations = concentrations.copy()
            concentrations[0] = 1e-9

        concentrations = interpolate_sub_horm(
            concentrations, subhormesis_index, hormesis_index, n_new, log=True
        )
        if subhormesis_index == 0:
            concentrations[0] = 0

        survival_rate = interpolate_sub_horm(
            survival_rate, subhormesis_index, hormesis_index, n_new, log=False
        )

        hormesis_index += n_new

    tox_survival = survival_rate.copy()
    tox_survival[0] = 1.0

    filtered_tox_survival = np.concatenate(
        (tox_survival[:1], tox_survival[hormesis_index:])
    )
    filtered_concentrations = np.concatenate(
        (concentrations[:1], concentrations[hormesis_index:])
    )

    tox_fit_params = lbfgs_fit(
        filtered_concentrations.tolist(),
        filtered_tox_survival.tolist(),
        b=None,
        c=0,
        d=1,
        e=None,
    )
    cleaned_tox_func = np.vectorize(fix_wlb1(tox_fit_params))

    if hormesis_index > 1:
        tox_survival[1:hormesis_index] = cleaned_tox_func(
            concentrations[1:hormesis_index]
        )

    return (
        concentrations,
        survival_rate,
        tox_survival,
        cleaned_tox_func,
        hormesis_index,
        tox_fit_params,
    )


def get_hormesis_free_model(
    data: CauseEffectData,
    max_survival: float,
    hormesis_index: Optional[int],
    interpolate: bool = True,
) -> Callable:
    (
        concentrations,
        survival_rate,
        tox_survival,
        cleaned_tox_func,
        hormesis_index,
        tox_fit_params,
    ) = fit_hormesis_free_response(
        data=data,
        max_survival=max_survival,
        hormesis_index=hormesis_index,
        interpolate=interpolate,
    )
    return cleaned_tox_func
