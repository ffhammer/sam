import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import beta
from icecream import ic
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Callable
import warnings
from scipy.optimize import brentq


@dataclass
class StandardSettings:

    beta_q: float = 3.2
    beta_p: float = 3.2
    survival_max: float = 100
    len_curves = 10_000


@dataclass
class ModelPredictions:
    concentration_curve: np.ndarray
    survival_curve: np.array
    stress_curve: np.array
    predicted_survival: np.array
    optim_param: np.array
    model: Callable
    lc1: float
    lc99: float
    hormesis_index : int


def dose_response_fit(
    concentration: np.ndarray,
    survival_observerd: np.ndarray,
    hormesis_concentration: Optional[float] = None,
    cfg: StandardSettings = StandardSettings(),
) -> ModelPredictions:

    survival_observerd[0] = cfg.survival_max

    # general value checks
    if cfg.survival_max <= 0:
        raise ValueError("survival_max must be >= 0")
    if len(concentration) != len(survival_observerd):
        raise ValueError(
            "concentration and survival_observerd must have the same length."
        )
    if len(concentration) > len(set(concentration)):
        raise ValueError("Concentrations must be unique.")
    if (np.sort(concentration) != concentration).all():
        raise ValueError("The concentration values must be in sorted order.")
    if any(np.array(concentration) < 0):
        raise ValueError("Concentrations must be >= 0")
    if min(concentration) > 0:
        raise ValueError("No control is given. The first concentration must be 0.")

    if not isinstance(concentration, np.ndarray) or concentration.dtype != np.float64:
        warnings.warn("Casting concentration to np.float64 array")
        concentration = np.array(concentration, np.float64)

    if (
        not isinstance(survival_observerd, np.ndarray)
        or survival_observerd.dtype != np.float64
    ):
        warnings.warn("Casting survival_observerd to np.float64 array")
        survival_observerd = np.array(survival_observerd, np.float64)

    if any(survival_observerd > cfg.survival_max) or any(survival_observerd < 0):
        raise ValueError("Observed survival must be between 0 and survival_max.")

    regress_conc, regress_surv, hormesis_index = get_regression_data(
        orig_concentration=concentration,
        orig_survival_observerd=survival_observerd,
        hormesis_concentration=hormesis_concentration,
        cfg=cfg,
    )

    fitted_func, optim_param = fit_weibull(
        concentration=regress_conc, survival=regress_surv
    )

    return compute_predictions(
        model=fitted_func, optim_param=optim_param, concentration=concentration, cfg=cfg, hormesis_index = hormesis_index,
    )


def compute_lc(model, lc: int, min_val: float, max_val: float):

    val = 1 - lc / 100

    def func(x):

        return model(x) - val

    if func(min_val) < 0:
        return min_val

    return brentq(func, min_val, max_val)


def compute_predictions(
    model, optim_param: np.array, concentration: np.array, cfg: StandardSettings, hormesis_index : int
):

    min_val = 1e-9
    max_val = find_lc_99_max(model)
    

    lc1 = compute_lc(model=model, lc=1, max_val=max_val, min_val=min_val)
    lc99 = compute_lc(model=model, lc=99, max_val=max_val, min_val=min_val)
    
    padded_concentration = pad_controll_concentration(concentration)

    concentration_curve = 10 ** np.linspace(
        np.log10(padded_concentration[0]), np.log10(concentration.max()), cfg.len_curves
    )
    pred_survival = model(concentration_curve)
    survival_curve = cfg.survival_max * pred_survival
    stress_curve = survival_to_stress(pred_survival, p=cfg.beta_p, q=cfg.beta_q)
    predicted_survival = model(padded_concentration)


    return ModelPredictions(
        concentration_curve=concentration_curve,
        survival_curve=survival_curve,
        stress_curve=stress_curve,
        predicted_survival=predicted_survival,
        optim_param=optim_param,
        model=model,
        lc1=lc1,
        lc99=lc99,
        hormesis_index=hormesis_index
    )


def find_lc_99_max(func):

    x = 10.0

    while not func(x) < 0.01:

        x *= 2

    return x


def pad_controll_concentration(orig_concentration: np.array) -> np.array:
    concentration = orig_concentration.copy()
    min_conc = 10 ** np.floor(np.log10(concentration[1]) - 2)
    concentration[0] = min_conc
    return concentration


def get_regression_data(
    orig_concentration: np.ndarray,
    orig_survival_observerd: np.ndarray,
    hormesis_concentration: Optional[float] = None,
    cfg: StandardSettings = StandardSettings(),
) -> Tuple[np.ndarray, np.ndarray]:

    concentration = pad_controll_concentration(orig_concentration=orig_concentration)

    survival = orig_survival_observerd / cfg.survival_max

    if hormesis_concentration is not None:

        if hormesis_concentration not in concentration:
            raise ValueError(
                "hormesis_concentration must equal one of the concentration values."
            )

        hormesis_index = np.where(hormesis_concentration == concentration)[0][0]

        if hormesis_index < 1 or hormesis_index + 1 >= len(concentration):
            raise ValueError(
                "hormesis_concentration must correspond to a index between the first and last position"
            )

        # skip sub hormesis values
        concentration = np.concatenate(
            (concentration[:2], concentration[hormesis_index:])
        )
        survival = np.concatenate((survival[:2], survival[hormesis_index:]))

    return concentration, survival, hormesis_index


def fit_weibull(concentration, survival):
    def weibull(x, b, e):
        return np.exp(-np.exp(b * (np.log(x) - e)))

    bounds = ([1e-7, 1e-7], [np.inf, np.inf])

    popt, _ = curve_fit(weibull, concentration, survival, bounds=bounds)
    return lambda x: weibull(x, *popt), popt


def clamp(x, lower=0, upper=1):
    """
    Clamps the values in x to be within the interval [lower, upper].
    """
    return np.minimum(np.maximum(x, lower), upper)


def survival_to_stress(survival, p=3.2, q=3.2):
    """
    Converts survival rates to stress values using the beta distribution CDF.

    Parameters:
    - survival: array-like, survival rates
    - p: shape parameter p of the beta distribution
    - q: shape parameter q of the beta distribution

    Returns:
    - stress: array-like, stress values
    """
    if p < 0 or q < 0:
        raise ValueError("Parameters p and q must be non-negative.")

    survival = clamp(survival)
    return beta.ppf(1 - survival, p, q)
