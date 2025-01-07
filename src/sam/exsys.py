from _warnings import warn
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from dataclasses_json import dataclass_json
from py_lbfgs import fix_wlb1, lbfgs_fit, wbl1_params

from .data_formats import CauseEffectData
from .helpers import detect_hormesis_index, pad_c0
from .stress_survival_conversion import stress_to_survival, survival_to_stress


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


def exsys(
    data: CauseEffectData,
    max_survival: float,
    hormesis_index: Optional[int] = None,
    len_curves: int = 1000,
    beta_q=3.2,
    beta_p=3.2,
    interpolate=True,
):
    if hormesis_index is None:
        warn("Try to detect hormesis automatically")
        hormesis_index = detect_hormesis_index(data.survival_rate)

    if (
        hormesis_index is None
        or hormesis_index < 1
        or hormesis_index >= len(data.concentration)
    ):
        raise ValueError("hormeis index must be  0 < index < len(data)")

    subhormesis_index = hormesis_index - 1

    if max_survival < 0:
        raise ValueError("Max Survival must be >= 0")

    concentrations = data.concentration

    survival_rate = data.survival_rate / max_survival

    min_conc = pad_c0(data.concentration)[0]
    print(concentrations, min_conc)

    keep = np.ones(len(concentrations), dtype=bool)
    if interpolate:
        # interpolate between subhormesis and hormesis
        n_new = 3
        print(len(concentrations))
        concentrations = interpolate_sub_horm(
            concentrations, subhormesis_index, hormesis_index, n_new, log=True
        )
        survival_rate = interpolate_sub_horm(
            survival_rate,
            subhormesis_index,
            hormesis_index,
            n_new,
            log=False,
        )
        keep = np.concatenate(
            (
                keep[: subhormesis_index + 1],
                np.array([False] * n_new),
                keep[hormesis_index:],
            )
        )
        assert len(keep) == len(concentrations)
        assert (concentrations[keep] == data.concentration).all()
        hormesis_index = hormesis_index + n_new

    tox_survival = survival_rate.copy()
    tox_survival[0] = 1.0

    # drop all parameters between hormesis and first point
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

    observed_stress = survival_to_stress(survival_rate, p=beta_p, q=beta_q)
    tox_stress = survival_to_stress(tox_survival, p=beta_p, q=beta_q)

    sys_stress_points = np.clip(observed_stress - tox_stress, 0, 1)

    if np.all(sys_stress_points == 0):
        raise ValueError("All sys stress points are 0")

    sys_stress_params = lbfgs_fit(
        tox_stress.tolist(), sys_stress_points.tolist(), b=None, c=0, d=None, e=None
    )
    sys_stress_func = np.vectorize(fix_wlb1(sys_stress_params))

    sys_stress = sys_stress_func(np.maximum(tox_stress, 1e-8))

    if any(sys_stress < 0):
        raise ValueError(
            "Something went wrong. The predicted system stress should not be < 0"
        )

    tox_sys_stress = tox_stress + sys_stress
    tox_sys_survival = stress_to_survival(tox_sys_stress, p=beta_p, q=beta_q)

    # creating curves
    concentrations_smooth = np.logspace(
        np.log10(min_conc), np.log10(concentrations.max()), len_curves
    )
    tox_survival_smooth = cleaned_tox_func(concentrations_smooth)
    tox_stress_smooth = survival_to_stress(tox_survival_smooth, p=beta_p, q=beta_q)
    sys_stress_smooth = sys_stress_func(tox_stress_smooth)
    tox_sys_stress_smooth = tox_stress_smooth + sys_stress_smooth
    tox_sys_survival_smooth = stress_to_survival(
        tox_sys_stress_smooth, p=beta_p, q=beta_q
    )

    return ExSysOutput(
        input_data=data,
        hormesis_index=hormesis_index,
        concentration=concentrations_smooth,
        tox_survival=tox_survival_smooth,
        tox_stress=tox_stress_smooth,
        sys_stress=sys_stress_smooth,
        tox_sys_stress=tox_sys_stress_smooth,
        tox_sys_survival=tox_sys_survival_smooth,
        beta_p=beta_p,
        beta_q=beta_q,
        tox_surv_params=tox_fit_params._asdict(),
        sys_stress_params=sys_stress_params._asdict(),
        max_survival=max_survival,
    )


@dataclass
class ExSysOutput:
    input_data: CauseEffectData
    hormesis_index: int
    concentration: np.ndarray
    tox_survival: np.ndarray
    tox_stress: np.ndarray
    sys_stress: np.ndarray
    tox_sys_stress: np.ndarray
    tox_sys_survival: np.ndarray

    beta_q: float
    beta_p: float
    tox_surv_params: dict[str, float]
    sys_stress_params: dict[str, float]
    max_survival: float
