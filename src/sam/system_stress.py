import os
from dataclasses import dataclass
from typing import Optional
from warnings import warn

import numpy as np
from dataclasses_json import dataclass_json
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from .concentration_response_fits import (
    ConcentrationResponsePrediction,
    CRF_Settings,
    concentration_response_fit,
)
from .data_formats import CauseEffectData, ExperimentMetaData
from .helpers import detect_hormesis_index, pad_c0, weibull_2param, weibull_3param
from .io import make_np_config
from .plotting import SCATTER_SIZE
from .hormesis_free_response_fitting import fit_hormesis_free_response
from .stress_addition_model import (
    STANDARD_SAM_SETTING,
    SAM_Settings,
    SAMPrediction,
    generate_sam_prediction,
)


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


@dataclass_json
@dataclass
class SysAdjustedSamPrediction:
    #: The unmodified input cause-effect data.
    original_series: CauseEffectData

    #: The corresponding unmodified concentration-response fit.
    original_fit: ConcentrationResponsePrediction

    #: Additional system-level stress manually added
    additional_stress: float

    #: Index of original_series.survival_rate indicating the hormesis point, either provided or determined automatically.
    hormesis_index: int

    #: Predicted system stress values after adjustment for hormesis of the input concentration series.
    predicted_system_stress: np.ndarray = make_np_config()

    #: Resulting SAM prediction after system adjustment.
    result: SAMPrediction

    #:
    hormesis_free_model_parameter: dict[str, float]

    def plot(self, title: Optional[str] = None) -> Figure:
        """
        Plots the adjusted SAM prediction alongside the original fit and data.

        Parameters:
            title (Optional[str]): Title for the plot.

        Returns:
            Figure: A matplotlib figure containing the plot.
        """
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

    def save_to_file(self, file_path: str) -> None:
        """
        Saves the SysAdjustedSamPrediction to a JSON file.

        Parameters:
            file_path (str): The file path to save the JSON data.
        """
        with open(file_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, file_path: str) -> "SysAdjustedSamPrediction":
        """
        Loads a SysAdjustedSamPrediction from a JSON file.

        Parameters:
            file_path (str): The file path to load the JSON data from.

        Returns:
            SysAdjustedSamPrediction: The loaded SysAdjustedSamPrediction.
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"Can't find file at {file_path}")

        with open(file_path, "r") as f:
            return cls.from_json(f.read())


def generate_sys_adjusted_sam_prediction(
    control_data: CauseEffectData,
    co_stressor_data: CauseEffectData,
    additional_stress: float,
    meta: Optional[ExperimentMetaData] = None,
    max_survival: Optional[float] = None,
    settings: SAM_Settings = STANDARD_SAM_SETTING,
    hormesis_index: Optional[int] = None,
) -> SysAdjustedSamPrediction:
    """
    Generates a system-adjusted SAM prediction by modifying the control data's survival rate to account for system stress.

    This function adjusts the control data to remove inherent system stress and replaces it with a manually specified
    value (`additional_stress`). The adjustment process includes:

    1. **Concentration-Response Fit on Adjusted Data**:
       - Processes the input data to ignore sub-hormesis effects and sets the control survival rate (at concentration=0)
         to 100%.
       - Performs a concentration-response fit to approximate the effect without the inherent system stress.

    2. **Prediction Without System Stress**:
       - Predicts survival rates for the control_date.concentration values using the "cleaned" model (without system stress).

    3. **Conversion to Stress, adding Additional Stress and Back to Survival**:
       - Converts survival predictions to stress values, adds the manual `additional_stress`, and converts back to
         survival rates.

    4. **SAM Prediction**:
       - Updates the adjusted control survival data and generates a new SAM prediction using the modified data.

    Parameters:
        control_data (CauseEffectData): The original control group concentration-response data.
        co_stressor_data (CauseEffectData): The co-stressor concentration-response data.
        additional_stress (float): The manually specified additional system stress to apply.
        meta (Optional[ExperimentMetaData]): Metadata for the experiment. Used to infer `max_survival` if not provided.
        max_survival (Optional[float]): The maximum survival rate. Overrides `meta.max_survival` if specified.
        settings (SAM_Settings): Configuration settings for the SAM prediction.
        hormesis_index (Optional[int]): Index of the hormesis point. If not provided, it will be detected automatically.

    Returns:
        SysAdjustedSamPrediction: The adjusted SAM prediction, including the modified control data, original fits, and
        predicted system stress values.

    Example:
        ```python
        prediction = generate_sys_adjusted_sam_prediction(
            control_data=control,
            co_stressor_data=co_stressor,
            additional_stress=0.2,
            meta=experiment_meta,
            settings=custom_settings,
        )
        ```
    """
    if hormesis_index is None:
        warn("Try to detect hormesis automatically")
        hormesis_index = detect_hormesis_index(control_data.survival_rate)

    if (
        hormesis_index is None
        or hormesis_index < 0
        or hormesis_index >= len(control_data.concentration)
    ):
        raise ValueError("Invalid hormesis index")

    if max_survival is None and meta is None:
        raise ValueError(
            "Either `max_survival` or `meta` with a defined `max_survival` must be provided. "
            "Specify `meta` or directly set `max_survival` to proceed."
        )
    max_survival = meta.max_survival if max_survival is None else max_survival

    crf_cfg = CRF_Settings(
        max_survival=max_survival,
        param_d_norm=settings.normalize_survival_for_stress_conversion,
        beta_q=settings.beta_q,
        beta_p=settings.beta_p,
        curve_fit_lib=settings.curve_fit_lib,
    )

    old_fit: ConcentrationResponsePrediction = concentration_response_fit(
        control_data, crf_cfg
    )

    _, _, _, fitted_model_without_hormesis, _, hormesis_free_params = (
        fit_hormesis_free_response(
            data=control_data,
            max_survival=max_survival,
            hormesis_index=hormesis_index,
            interpolate=True,
        )
    )
    cleaned_survival = fitted_model_without_hormesis(
        pad_c0(control_data.concentration)
    )  # pad to deal with log of 0 warning, will replace surv[0] anyways

    with_add_stress = (
        settings.stress_to_survival(
            settings.survival_to_stress(cleaned_survival) + additional_stress
        )
        * max_survival
    )

    with_add_stress[0] = control_data.survival_rate[0]

    new_main_series = CauseEffectData(
        control_data.concentration,
        survival_rate=with_add_stress,
        name=f"{control_data.name}_with_add_stress_{additional_stress}",
    )

    prediction: SAMPrediction = generate_sam_prediction(
        control_data=new_main_series,
        co_stressor_data=co_stressor_data,
        meta=meta,
        max_survival=max_survival,
        settings=settings,
    )

    predicted_system_stress = settings.survival_to_stress(
        fitted_model_without_hormesis(prediction.control.concentration)
    )

    return SysAdjustedSamPrediction(
        original_series=control_data,
        original_fit=old_fit,
        result=prediction,
        additional_stress=additional_stress,
        hormesis_index=hormesis_index,
        predicted_system_stress=predicted_system_stress,
        hormesis_free_model_parameter=hormesis_free_params._asdict(),
    )
