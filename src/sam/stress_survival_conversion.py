import numpy as np
from scipy.stats import beta

def survival_to_stress(survival: np.ndarray, p: float = 3.2, q: float = 3.2) -> np.ndarray:
    """
    Converts survival rates to stress values using a beta distribution.
    """
    if p < 0 or q < 0:
        raise ValueError("Parameters p and q must be non-negative.")

    survival = np.clip(survival, 0, 1)
    return beta.ppf(1 - survival, p, q)


def stress_to_survival(stress, p=3.2, q=3.2):
    """
    Converts stress values to survival rates using a beta distribution.
    """
    if p < 0 or q < 0:
        raise ValueError("Parameters p and q must be non-negative.")
    
    return 1 - beta.cdf(stress, p, q)