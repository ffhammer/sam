from functools import partial
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
from py_lbfgs import wbl1_params


def ll5(conc, b, c, d, e, f):
    return c + (d - c) / (1 + (conc / e) ** b) ** f


def ll5_inv(surv, b, c, d, e, f):
    return e * (((d - c) / (surv - c)) ** (1 / f) - 1) ** (1 / b)


def weibull_2param(x, b, e):
    return np.exp(-np.exp(b * (np.log(x) - np.log(e))))


def weibull_2param_inverse(y, b, e):
    return np.exp((np.log(-np.log(y)) / b) + np.log(e))


def weibull_3param(x, b, d, e):
    return d * np.exp(-np.exp(b * (np.log(x) - np.log(e))))


def wlb1(x: float, b: float, c: float, d: float, e: float) -> float:
    """
    Compute the Weibull function value for a given x and parameters b, c, d, e.

    Args:
        x (float): Independent variable value.
        b (float): Parameter b.
        c (float): Parameter c.
        d (float): Parameter d.
        e (float): Parameter e.

    Returns:
        float: The value of the Weibull function at x.
    """
    return c + (d - c) * np.exp(-np.exp(b * (np.log(x) - np.log(e))))


def fix_wlb1(params: wbl1_params) -> Callable:
    """
    Create a partially applied Weibull function with some fixed parameters.

    Args:
        params (wbl1_params): Parameters to fix in the Weibull function.

    Returns:
        Callable: A Weibull function with fixed parameters.
    """
    return partial(wlb1, b=params.b, c=params.c, d=params.d, e=params.e)


def compute_control_addition_lc(
    control_params: dict[str, float],
    co_stressor_params: dict[str, float],
    lc: float,
) -> float:
    pa = control_params
    pb = co_stressor_params

    val = (1 - (lc / 100)) * pb["d"]

    conc_env_ca = pa["e"] * (
        ((pa["d"] / pb["d"]) ** (1 / pa["f"]) - 1) ** (1 / pa["b"])
    )

    return ll5_inv(val, **control_params) - conc_env_ca


@dataclass
class Predicted_LCs:
    stress_lc10: float
    stress_lc50: float
    sam_lc10: float
    sam_lc50: float


def detect_hormesis_index(survival_series) -> Optional[int]:
    diff = survival_series[:-1] - survival_series[1:]

    if not (diff < 0).any():
        return None

    index = np.argmin(diff) + 1

    if index == len(survival_series) - 1:
        return None

    val = index

    for i in range(index + 1, len(survival_series) - 1):
        if survival_series[i] > survival_series[index - 1]:
            val = i

    return val


def pad_c0(orig_concentration: np.array) -> np.array:
    """
    Pads the control concentration value.

    Args:
        orig_concentration (np.array): Original concentration values.

    Returns:
        np.array: Padded concentration values.
    """
    concentration = orig_concentration.copy()
    min_conc = 10 ** np.floor(np.log10(concentration[1]) - 2)
    concentration[0] = min_conc
    return concentration


def compute_lc(optim_param: dict[str, float], lc: int) -> float:
    """
    Computes the lethal concentration for a given percentage of the population.

    Args:
        model (Callable): The fitted model.
        lc (int): Lethal concentration percentage.
        min_val (float): Minimum value for the root-finding algorithm.
        max_val (float): Maximum value for the root-finding algorithm.

    Returns:
        float: The computed lethal concentration.
    """

    c0 = optim_param["d"]

    if c0 < 0 or c0 > 1:
        raise ValueError("c0 must be between 0 and 1")

    frac = 1 - lc / 100

    val = frac * c0

    return ll5_inv(val, **optim_param)
