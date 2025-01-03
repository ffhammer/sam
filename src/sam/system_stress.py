from warnings import warn
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from matplotlib.figure import Figure

from .io import make_np_config
from typing import Optional
from .data_formats import ExperimentMetaData
from .stress_addition_model import (
    generate_sam_prediction,
    SAMPrediction,
    STANDARD_SAM_SETTING,
    SAM_Settings,
)
from .data_formats import CauseEffectData
from .concentration_response_fits import (
    CRF_Settings,
    ConcentrationResponsePrediction,
    concentration_response_fit,
)
from .helpers import pad_c0, weibull_2param, weibull_3param, detect_hormesis_index
from .plotting import SCATTER_SIZE


def fallback_linear_regression(x_data, y_data):
    reg = LinearRegression()
    reg.fit(np.log(x_data).reshape(-1, 1), y_data)  # log-transformed linear regression
    return lambda x: reg.predict(np.log(x).reshape(-1, 1))


def fit_weibull_2param(x_data, y_data):
    initial_guess = [1, 1]
    param_bounds = ([-20, 1e-8], [20, 1e5])

    try:
        # Try to fit Weibull model
        popt, pcov = curve_fit(
            weibull_2param, x_data, y_data, p0=initial_guess, bounds=param_bounds
        )
        return lambda x: weibull_2param(x, *popt), popt
    except Exception as e:
        warn(f"Weibull 2-param fit failed wiht {e}, defaulting to linear regression")
        return fallback_linear_regression(x_data, y_data)


def fit_weibull_3param(x_data, y_data):
    initial_guess = [1, 1, 1]  # Initial guesses for b, d, e
    param_bounds = ([0.05, 1e-8, 1e-8], [3, 1e5, 1e5])

    try:
        # Try to fit Weibull model
        popt, pcov = curve_fit(
            weibull_3param, x_data, y_data, p0=initial_guess, bounds=param_bounds
        )
        return lambda x: weibull_3param(x, *popt)
    except Exception as e:
        warn(f"Weibull 3-param fit with {e}, defaulting to linear regression")
        return fallback_linear_regression(x_data, y_data)


def pred_surv_without_hormesis(concentration, surv_withhormesis, hormesis_index):
    survival_cleaned = np.concatenate(([1.0], surv_withhormesis[hormesis_index:]))

    concentration_cleaned = np.concatenate(
        (concentration[:1], concentration[hormesis_index:])
    )

    fitted_func, popt = fit_weibull_2param(
        x_data=concentration_cleaned, y_data=survival_cleaned
    )

    return fitted_func, popt


@dataclass_json
@dataclass
class CleanedPred:
    original_series: CauseEffectData
    original_fit: ConcentrationResponsePrediction
    additional_stress: float
    hormesis_index: int
    predicted_system_stress: np.ndarray = make_np_config()
    result: SAMPrediction

    def plot(self, title: Optional[str] = None) -> Figure:
        fig = self.result.plot(title=title)
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]

        ax1.plot(
            self.result.control.concentration,
            self.original_fit.survival,
            label="Original",
            linestyle="--",
            c="black",
        )
        ax2.plot(
            self.result.control.concentration,
            self.original_fit.general_stress,
            label="Original",
            linestyle="--",
            c="black",
        )

        color = [
            "black" if i != self.hormesis_index else "red"
            for i in range(len(self.original_series.concentration))
        ]

        ax1.scatter(
            pad_c0(self.original_series.concentration),
            self.original_series.survival_rate,
            label="Original",
            c=color,
            s=SCATTER_SIZE,
        )
        ax2.legend()
        return fig


def predict_with_hormesis_cancelled(
    main_series: CauseEffectData,
    stressor_series: CauseEffectData,
    additional_stress: float,
    meta: Optional[ExperimentMetaData] = None,
    max_survival: Optional[float] = None,
    settings: SAM_Settings = STANDARD_SAM_SETTING,
    hormesis_index: Optional[int] = None,
) -> CleanedPred:
    if hormesis_index is None:
        warn("Try to detect hormesis automatically")
        hormesis_index = detect_hormesis_index(main_series.survival_rate)

    if (
        hormesis_index is None
        or hormesis_index < 0
        or hormesis_index >= len(main_series.concentration)
    ):
        raise ValueError("Invalid hormesis index")

    if max_survival is None and meta is None:
        raise ValueError(
            "Either `max_survival` or `meta` with a defined `max_survival` must be provided. "
            "Specify `meta` or directly set `max_survival` to proceed."
        )
    max_survival = meta.max_survival if max_survival is None else max_survival

    dose_cfg = CRF_Settings(
        max_survival=max_survival,
        param_d_norm=settings.normalize_survival_for_stress_conversion,
        beta_q=settings.beta_q,
        beta_p=settings.beta_p,
        curve_fit_lib=settings.curve_fit_lib,
    )

    old_fit: ConcentrationResponsePrediction = concentration_response_fit(
        main_series, dose_cfg
    )

    fitted_model_without_hormesis, _ = pred_surv_without_hormesis(
        pad_c0(main_series.concentration),
        main_series.survival_rate / max_survival,
        hormesis_index=hormesis_index,
    )
    cleaned_survival = fitted_model_without_hormesis(
        pad_c0(main_series.concentration)
    )  # pad to deal with log of 0 warning, will replace surv[0] anyways

    with_add_stress = (
        settings.stress_to_survival(
            settings.survival_to_stress(cleaned_survival) + additional_stress
        )
        * max_survival
    )

    with_add_stress[0] = main_series.survival_rate[0]

    new_main_series = CauseEffectData(
        main_series.concentration,
        survival_rate=with_add_stress,
        name=f"{main_series.name}_with_add_stress_{additional_stress}",
    )

    prediction: SAMPrediction = generate_sam_prediction(
        control=new_main_series,
        co_stressor=stressor_series,
        meta=meta,
        max_survival=max_survival,
        settings=settings,
    )

    predicted_system_stress = settings.survival_to_stress(
        fitted_model_without_hormesis(prediction.control.concentration)
    )

    return CleanedPred(
        original_series=main_series,
        original_fit=old_fit,
        result=prediction,
        additional_stress=additional_stress,
        hormesis_index=hormesis_index,
        predicted_system_stress=predicted_system_stress,
    )
