import numpy as np
from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path


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
    
    
def compute_lc_from_curve(concentrations : np.ndarray, survival_curve: np.ndarray, lc : float, survival_max : float, c0 : float):
    
    
    normed = survival_curve / survival_max
    
    val = 1 - (lc / 100)
    val *= c0
    
    arg = np.argmax(normed < val)
    
    if arg == 0: 
        return np.nan
    
    return float(concentrations[arg])
    

@dataclass
class Predicted_LCs:
    stress_lc10 : float
    stress_lc50 : float
    sam_lc10 : float
    sam_lc50 : float


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



def compute_lc(optim_param : dict[str, float], lc: int) -> float:
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


