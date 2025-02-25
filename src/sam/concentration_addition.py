from .helpers import ll5_inv, ll5, LC_Verification_Error
import numpy as np


def concentration_addition_prediction(
    control_params: dict[str, float],
    co_stressor_params: dict[str, float],
    concentration: np.ndarray,
    max_survival: float,
) -> np.ndarray:
    pa = control_params
    pb = co_stressor_params

    conc_env_ca = ll5_inv(min(pb["d"], pa["d"] * 0.999), **pa)

    return ll5(concentration + conc_env_ca, **pa) * max_survival


def compute_control_addition_lc(
    control_params: dict[str, float],
    co_stressor_params: dict[str, float],
    lc: float,
) -> float:
    pa = control_params
    pb = co_stressor_params

    val = (1 - (lc / 100)) * pb["d"]

    # do the min to provide numerical stability for the edgecase, where pa["d"] <= pb["d"]
    conc_env_ca = ll5_inv(min(pb["d"], pa["d"] * 0.999), **pa)

    res = ll5_inv(val, **pa) - conc_env_ca

    revalidated = concentration_addition_prediction(
        control_params=control_params,
        co_stressor_params=co_stressor_params,
        concentration=res,
        max_survival=1,
    )
    if not np.isclose(val, revalidated):
        raise LC_Verification_Error(
            f"Inverse mismatch: expected {val}, got {revalidated}; "
            f"res: {res}, control_params: {pa}, co_stressor_params: {pb}"
        )
    return res
