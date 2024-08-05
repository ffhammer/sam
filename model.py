import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import beta
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import warnings
from scipy.optimize import brentq

@dataclass
class StandardSettings:
    """
    Contains default settings for the model.

    Attributes:
        beta_q (float): Beta distribution shape parameter for stress computation.
        beta_p (float): Beta distribution shape parameter for stress computation.
        survival_max (float): Maximum observed survival.
        len_curves (int): Length of the survival curve.
    """
    beta_q: float = 3.2
    beta_p: float = 3.2
    survival_max: float = 100
    len_curves = 10_000

@dataclass
class ModelInputs:
    """
    Holds the input data required for the dose-response model.

    Attributes:
        concentration (np.ndarray): Array of concentration values.
        survival_observered (np.ndarray): Array of observed survival values.
        hormesis_concentration (Optional[float]): Concentration value at which hormesis occurs.
        cfg (StandardSettings): Configuration settings.
    """
    concentration: np.ndarray
    survival_observered: np.ndarray
    hormesis_concentration: Optional[float]
    cfg: StandardSettings

@dataclass
class ModelPredictions:
    """
    Stores the results of the model predictions.

    Attributes:
        concentration_curve (np.ndarray): Concentration values for the fitted curve.
        survival_curve (np.array): Predicted survival values.
        stress_curve (np.array): Stress values computed from the survival data.
        predicted_survival (np.array): Predicted survival values for the input concentrations.
        optim_param (np.array): Optimized parameters of the Weibull model.
        model (Callable): The fitted Weibull model.
        lc1 (float): Lethal concentration for 1% of the population.
        lc99 (float): Lethal concentration for 99% of the population.
        hormesis_index (int): Index of the hormesis concentration.
        inputs (ModelInputs): The inputs provided to the model.
    """
    concentration_curve: np.ndarray
    survival_curve: np.array
    stress_curve: np.array
    predicted_survival: np.array
    optim_param: np.array
    model: Callable
    lc1: float
    lc99: float
    hormesis_index: int
    inputs : ModelInputs

def dose_response_fit(
    concentration: np.ndarray,
    survival_observerd: np.ndarray,
    hormesis_concentration: Optional[float] = None,
    cfg: StandardSettings = StandardSettings(),
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
        concentration (np.ndarray): Array of concentration values.
        survival_observerd (np.ndarray): Array of observed survival values.
        hormesis_concentration (Optional[float], optional): Concentration value at which hormesis occurs. Defaults to None.
        cfg (StandardSettings, optional): Configuration settings. Defaults to StandardSettings().

    Returns:
        ModelPredictions: The fitted model predictions and related data.
    """

    if cfg.survival_max <= 0:
        raise ValueError("survival_max must be >= 0")
    if len(concentration) != len(survival_observerd):
        raise ValueError("concentration and survival_observerd must have the same length.")
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

    if not isinstance(survival_observerd, np.ndarray) or survival_observerd.dtype != np.float64:
        warnings.warn("Casting survival_observerd to np.float64 array")
        survival_observerd = np.array(survival_observerd, np.float64)

    if any(survival_observerd > cfg.survival_max) or any(survival_observerd < 0):
        raise ValueError("Observed survival must be between 0 and survival_max.")

    inputs = ModelInputs(
        concentration=concentration,
        survival_observered=survival_observerd,
        hormesis_concentration=hormesis_concentration,
        cfg=cfg,
    )
    
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
        model=fitted_func,
        optim_param=optim_param,
        inputs=inputs,
        cfg=cfg,
        hormesis_index=hormesis_index,
    )

def compute_lc(model, lc: int, min_val: float, max_val: float) -> float:
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
    val = 1 - lc / 100

    def func(x):
        return model(x) - val

    if func(min_val) < 0:
        return min_val

    return brentq(func, min_val, max_val)

def compute_predictions(
    model,
    optim_param: np.array,
    inputs : ModelInputs,
    cfg: StandardSettings,
    hormesis_index: int,
) -> ModelPredictions:
    """
    Computes the survival and stress predictions based on the fitted model.

    Args:
        model (Callable): The fitted Weibull model.
        optim_param (np.array): Optimized parameters.
        inputs (ModelInputs): The input data.
        cfg (StandardSettings): Configuration settings.
        hormesis_index (int): Index of the hormesis concentration.

    Returns:
        ModelPredictions: The model predictions.
    """
    min_val = 1e-9
    max_val = find_lc_99_max(model)

    lc1 = compute_lc(model=model, lc=1, max_val=max_val, min_val=min_val)
    lc99 = compute_lc(model=model, lc=99, max_val=max_val, min_val=min_val)

    padded_concentration = pad_controll_concentration(inputs.concentration)

    concentration_curve = 10 ** np.linspace(
        np.log10(padded_concentration[0]), np.log10(inputs.concentration.max()), cfg.len_curves
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
        hormesis_index=hormesis_index,
        inputs=inputs,
    )

def find_lc_99_max(func) -> float:
    """
    Finds the maximum concentration value for which the survival is less than 1%.

    Args:
        func (Callable): The fitted Weibull function.

    Returns:
        float: The maximum concentration value.
    """
    x = 10.0

    while not func(x) < 0.01:
        x *= 2

    return x

def pad_controll_concentration(orig_concentration: np.array) -> np.array:
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

def get_regression_data(
    orig_concentration: np.ndarray,
    orig_survival_observerd: np.ndarray,
    hormesis_concentration: Optional[float] = None,
    cfg: StandardSettings = StandardSettings(),
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Prepares the data for regression analysis, handling hormesis concentration if provided.

    Args:
        orig_concentration (np.ndarray): Original concentration values.
        orig_survival_observerd (np.ndarray): Original observed survival values.
        hormesis_concentration (Optional[float], optional): Hormesis concentration. Defaults to None.
        cfg (StandardSettings, optional): Configuration settings. Defaults to StandardSettings().

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: Prepared concentration and survival data, and hormesis index.
    """
    concentration = pad_controll_concentration(orig_concentration=orig_concentration)

    survival = orig_survival_observerd / cfg.survival_max
    survival[0] = 1

    if hormesis_concentration is not None:
        if hormesis_concentration not in concentration:
            raise ValueError("hormesis_concentration must equal one of the concentration values.")

        hormesis_index = np.where(hormesis_concentration == concentration)[0][0]

        if hormesis_index < 2 or hormesis_index + 1 >= len(concentration):
            raise ValueError("hormesis_concentration must correspond to an index between the first and last position")

        concentration = np.concatenate((concentration[:2], concentration[hormesis_index:]))
        survival = np.concatenate((survival[:2], survival[hormesis_index:]))

    else:
        hormesis_index = None

    return concentration, survival, hormesis_index

def fit_weibull(concentration: np.ndarray, survival: np.ndarray) -> Tuple[Callable, np.array]:
    """
    Fits a Weibull model to the data.

    Args:
        concentration (np.ndarray): Concentration values.
        survival (np.ndarray): Survival values.

    Returns:
        Callable: The fitted Weibull function.
        np.array: Optimized parameters.
    """
    def weibull(x, b, e):
        return np.exp(-np.exp(b * (np.log(x) - e)))

    bounds = ([1e-7, 1e-7], [np.inf, np.inf])

    popt, _ = curve_fit(weibull, concentration, survival, bounds=bounds)
    return lambda x: weibull(x, *popt), popt

def clamp(x: np.ndarray, lower: float = 0, upper: float = 1) -> np.ndarray:
    """
    Clamps the values in x to be within the interval [lower, upper].

    Args:
        x (np.ndarray): Input values.
        lower (float, optional): Lower bound. Defaults to 0.
        upper (float, optional): Upper bound. Defaults to 1.

    Returns:
        np.ndarray: Clamped values.
    """
    return np.minimum(np.maximum(x, lower), upper)

def survival_to_stress(survival: np.ndarray, p: float = 3.2, q: float = 3.2) -> np.ndarray:
    """
    Converts survival rates to stress values using the beta distribution CDF.

    Args:
        survival (np.ndarray): Survival rates.
        p (float, optional): Beta distribution shape parameter. Defaults to 3.2.
        q (float, optional): Beta distribution shape parameter. Defaults to 3.2.

    Returns:
        np.ndarray: Stress values.
    """
    if p < 0 or q < 0:
        raise ValueError("Parameters p and q must be non-negative.")

    survival = clamp(survival)
    return beta.ppf(1 - survival, p, q)
