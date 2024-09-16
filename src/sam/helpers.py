from scipy.optimize import brentq
import numpy as np
from dataclasses import dataclass

def compute_lc_from_curve(concentrations : np.ndarray, survival_curve: np.ndarray, llc : float, survival_max : float):
    
    
    normed = 1 - survival_curve / survival_max
    
    arg = np.argmax(normed > (llc / 100))
    
    return float(concentrations[arg])


@dataclass
class Predicted_LCs:
    stress_lc10 : float
    stress_lc50 : float
    sam_lc10 : float
    sam_lc50 : float

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

    return float(brentq(func, min_val, max_val))