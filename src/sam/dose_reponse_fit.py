import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple, Callable
import warnings
from .data_formats import DoseResponseSeries
from .stress_survival_conversion import survival_to_stress, stress_to_survival
from .helpers import compute_lc, ll5, ll5_inv, pad_c0
from .transforms import *

# Constants
CONC0_MAX_DY = 5.0 / 100
CONC0_MIN_EXP = -100


@dataclass
class FitSettings:
    """
    Contains default settings for the model.

    Attributes:
        survival_max (float): Maximum observed survival.
        len_curves (int): Length of the survival curve.
    """

    survival_max: float = 100
    len_curves: int = 10_000
    transform: Transforms = Transforms.williams_and_linear_interpolation
    param_d_norm: bool = False
    stress_to_survival: Callable = lambda x: stress_to_survival(x, 3.2, 3.2)
    survival_to_stress: Callable = lambda x: survival_to_stress(x, 3.2, 3.2)


@dataclass
class ModelPredictions:
    """
    Stores the results of the model predictions.

    Attributes:
        concentration_curve (np.ndarray): Concentration values for the fitted curve.
        survival_curve (np.array): Predicted survival values.
        stress_curve (np.array): Stress values computed from the survival data.
        predicted_survival (np.array): Predicted survival values for the input concentrations.
        model (Callable): The fitted Weibull model.
        lc1 (float): Lethal concentration for 1% of the population.
        lc99 (float): Lethal concentration for 99% of the population.
        hormesis_index (int): Index of the hormesis concentration.
        inputs (DoseResponseSeries): The inputs provided to the model.
        cfg (StandardSettings): Settings used
    """

    concentration_curve: np.ndarray
    survival_curve: np.array
    stress_curve: np.array
    predicted_survival: np.array
    optim_param: dict
    model: Callable
    lc1: float
    lc99: float
    inputs: DoseResponseSeries
    cfg: FitSettings


def dose_response_fit(
    dose_response_data: DoseResponseSeries,
    cfg: FitSettings = FitSettings(),
) -> ModelPredictions:
    """
    Fits a Weibull curve to the dose-response data and computes the stress curve using the SAM model.

    Important Assumptions:
        - The control (survival_observed[0]) is set to survival_max.
        - If hormesis is given, the data used for regression will be taken as
          concentration[:2] + concentration[hormesis_index:] and survival_observed[:2] + survival_observed[hormesis_index:],
          meaning the subhormesis range between the 2nd and hormesis_index data points is excluded.
          Therefore, hormesis_index must be at least in the 3rd position
          (e.g., regression_data = concentration[:2] + concentration[hormesis_index:] and survival[:2] + survival[hormesis_index:]).

    Args:
        cfg (StandardSettings, optional): Configuration settings. Defaults to StandardSettings().

    Returns:
        ModelPredictions: The fitted model predictions and related data.
    """

    concentration = dose_response_data.concentration
    survival_observerd = dose_response_data.survival_rate

    if cfg.survival_max <= 0:
        raise ValueError("survival_max must be >= 0")

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

    regress_conc, regress_surv = get_regression_data(
        orig_concentration=concentration,
        orig_survival_observerd=survival_observerd,
        cfg=cfg,
    )

    fitted_func, optim_param = fit_ll5(
        concentration=regress_conc, survival=regress_surv
    )

    return compute_predictions(
        model=fitted_func,
        optim_param=optim_param,
        inputs=dose_response_data,
        cfg=cfg,
    )



def compute_predictions(
    model,
    optim_param: np.array,
    inputs: DoseResponseSeries,
    cfg: FitSettings,
) -> ModelPredictions:
    """
    Computes the survival and stress predictions based on the fitted model.

    Args:
        model (Callable): The fitted Weibull model.
        optim_param (np.array): Optimized parameters.
        inputs (DoseResponseSeries): The input data.
        cfg (StandardSettings): Configuration settings.

    Returns:
        ModelPredictions: The model predictions.
    """

    lc1 = compute_lc(optim_param=optim_param, lc=1)
    lc99 = compute_lc(optim_param=optim_param, lc=99)

    padded_concentration = pad_c0(inputs.concentration)

    concentration_curve = 10 ** np.linspace(
        np.log10(padded_concentration[0]),
        np.log10(inputs.concentration.max()),
        cfg.len_curves,
    )
    pred_survival = model(concentration_curve)
    survival_curve = cfg.survival_max * pred_survival

    if cfg.param_d_norm:
        stress_curve = cfg.survival_to_stress(pred_survival / optim_param["d"])
    else:
        stress_curve = cfg.survival_to_stress(pred_survival)

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
        inputs=inputs,
        cfg=cfg,
    )




def get_regression_data(
    orig_concentration: np.ndarray,
    orig_survival_observerd: np.ndarray,
    cfg: FitSettings = FitSettings(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares the data for regression analysis, handling hormesis concentration if provided.

    Args:
        orig_concentration (np.ndarray): Original concentration values.
        orig_survival_observerd (np.ndarray): Original observed survival values.
        cfg (StandardSettings, optional): Configuration settings. Defaults to StandardSettings().

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: Prepared concentration and survival data, and hormesis index.
    """

    survival = orig_survival_observerd / cfg.survival_max

    transform_func = cfg.transform

    return transform_func(orig_concentration, survival)


def fit_ll5(
    concentration: np.ndarray, survival: np.ndarray
) -> Tuple[Callable, np.array]:

    fixed_params = {
        "c": 0,
        "d": survival[0],
    }

    bounds = {
        "b": [0, 100],
        "c": [0, max(survival)],
        "d": [0, 2 * max(survival)],
        "e": [0, max(concentration)],
        "f": [0.1, 10],
    }

    keep = {k: v for k, v in bounds.items() if k not in fixed_params}

    bounds_tup = tuple(zip(*keep.values()))

    def fitting_func(concentration, *args):

        params = fixed_params.copy()
        params.update({k: v for k, v in zip(keep.keys(), args)})
        return ll5(concentration, **params)

    popt, pcov = curve_fit(
        fitting_func,
        concentration,
        survival,
        p0=np.ones_like(bounds_tup[0]),
        bounds=bounds_tup,
    )

    params = fixed_params.copy()
    params.update({k: v for k, v in zip(keep.keys(), popt)})

    fitted_func = lambda conc: ll5(conc, **params)

    return fitted_func, params
