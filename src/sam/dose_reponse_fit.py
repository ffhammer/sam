import warnings
from dataclasses import dataclass, field
from typing import Callable, Tuple
import os

import numpy as np
from dataclasses_json import config, dataclass_json
from scipy.optimize import curve_fit

from .data_formats import DoseResponseSeries
from .helpers import compute_lc, ll5, pad_c0
from .io import make_np_config
from .stress_survival_conversion import stress_to_survival, survival_to_stress
from .transforms import Transforms

# Constants
CONC0_MAX_DY = 5.0 / 100
CONC0_MIN_EXP = -100


@dataclass_json
@dataclass
class DRF_Settings:
    """
    Configuration settings for dose-response fitting.

    These mappings between stress and survival allow control over how survival values are transformed
    in the fitting process.
    """

    #: Maximum possible survival rate, representing the control condition.
    max_survival: float = None

    #: Number of points in the generated curves for predictions (default is 10,000).
    len_curves: int = 10_000

    #: Transformation function applied to regression data before fitting.
    transform: "Transforms" = field(
        default=Transforms.williams_and_linear_interpolation,
        metadata=config(
            encoder=lambda t: t.__name__.replace("transform_", ""),
            decoder=lambda t: getattr(Transforms, t),
        ),
    )

    #: Parameter normalization setting (explained in `SAM_Settings`).
    param_d_norm: bool = False

    #: q Parameter of beta distribution for survival to stress and vice versa conversions
    beta_q: float = 3.2

    #: p Parameter of beta distribution for survival to stress and vice versa conversions
    beta_p: float = 3.2

    def __post_init__(
        self,
    ):
        self.stress_to_survival: Callable = lambda x: stress_to_survival(
            x, p=self.beta_p, q=self.beta_q
        )
        self.survival_to_stress: Callable = lambda x: survival_to_stress(
            x, p=self.beta_p, q=self.beta_q
        )


STANDARD_DRF_SETTING = DRF_Settings()

@dataclass_json
@dataclass
class ModelPredictions:
    """
    Contains the results of model predictions for dose-response data.

    This structure is used to store key components of the model output, allowing access to curves
    and lethal concentration metrics.
    """

    #: Concentration values for the fitted curve.
    concentrations: np.ndarray = make_np_config()

    #: Array of predicted survival values.
    survival_curve: np.ndarray = make_np_config()

    #: Array of computed stress values from survival data.
    stress_curve: np.ndarray = make_np_config()

    #: Array of predicted survival values for input concentrations.
    predicted_survival: np.ndarray = make_np_config()

    #: Dictionary of optimized parameters from the model.
    optim_param: dict

    #: Lethal concentration value for 1% of the population.
    lc1: float

    #: Lethal concentration value for 99% of the population.
    lc99: float

    #: Input series of dose-response data provided to the model.
    inputs: DoseResponseSeries

    #: Settings used for the dose-response fitting.
    cfg: DRF_Settings

    #: Concentration values after applying transformations (before fitting).
    regress_conc: np.ndarray = make_np_config()

    #: Survival values corresponding to `regress_conc`.
    regress_surv: np.ndarray = make_np_config()
    
    @property
    def model(self) -> Callable:
        
        return lambda conc: ll5(conc, **self.optim_param)
    
    def save_to_file(self, file_path : str) -> None:
        
        with open(file_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, file_path : str) -> None:
        
        if not os.path.isfile(file_path):
            raise ValueError(f"Can't find file at {file_path}")
    
        with open(file_path, "r") as f:
            return cls.from_json(f.read())


def dose_response_fit(
    dose_response_data: DoseResponseSeries,
    cfg: DRF_Settings = STANDARD_DRF_SETTING,
) -> ModelPredictions:
    """
    Fits a five-parameter log-logistic (LL5) model to dose-response data.

    Parameters:
        dose_response_data (DoseResponseSeries): Series of concentration and survival data points.
        cfg (DRF_Settings, optional): Configuration settings for dose-response fitting.
            Defaults to `STANDARD_DRF_SETTING`.

    Returns:
        ModelPredictions: Contains the fitted model's predictions, survival curve, stress curve,
        and lethal concentration values (e.g., LC1, LC99).

    This function takes dose-response data and fits an LL5 model to predict survival as a function
    of concentration, applying any specified transformations as needed.
    """

    if cfg.max_survival is None:
        if dose_response_data.meta is None:
            raise ValueError(
                "Either cfg.max_survival or dose_response_data.meta must be none None to infere the maximum Survival!"
            )
        cfg.survival_max = dose_response_data.meta.max_survival

    concentration = dose_response_data.concentration
    survival_observerd = dose_response_data.survival_rate

    if cfg.max_survival <= 0:
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

    if any(survival_observerd > cfg.max_survival) or any(survival_observerd < 0):
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
        regress_conc=regress_conc,
        regress_surv=regress_surv,
    )


def compute_predictions(
    model,
    optim_param: np.array,
    inputs: DoseResponseSeries,
    cfg: DRF_Settings,
    regress_conc: np.ndarray,
    regress_surv: np.ndarray,
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
    survival_curve = cfg.max_survival * pred_survival

    if cfg.param_d_norm:
        stress_curve = cfg.survival_to_stress(pred_survival / optim_param["d"])
    else:
        stress_curve = cfg.survival_to_stress(pred_survival)

    predicted_survival = model(padded_concentration)
    return ModelPredictions(
        concentrations=concentration_curve,
        survival_curve=survival_curve,
        stress_curve=stress_curve,
        predicted_survival=predicted_survival,
        optim_param=optim_param,
        lc1=lc1,
        lc99=lc99,
        inputs=inputs,
        cfg=cfg,
        regress_conc=regress_conc,
        regress_surv=regress_surv,
    )


def get_regression_data(
    orig_concentration: np.ndarray,
    orig_survival_observerd: np.ndarray,
    cfg: DRF_Settings = DRF_Settings(),
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

    survival = orig_survival_observerd / cfg.max_survival

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
