import numpy as np
from scipy.stats import beta


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


def stress_to_survival(stress, p=3.2, q=3.2):
    """
    Converts stress values to survival rates using the beta distribution CDF.

    Args:
        stress (float or array-like): Stress values.
        p (float, optional): Shape parameter p of the beta distribution. Defaults to 3.2.
        q (float, optional): Shape parameter q of the beta distribution. Defaults to 3.2.

    Returns:
        float or array-like: Survival rates.
    """
    if p < 0 or q < 0:
        raise ValueError("Parameters p and q must be non-negative.")
    
    return 1 - beta.cdf(stress, p, q)