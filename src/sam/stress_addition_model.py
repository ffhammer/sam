from dataclasses import dataclass, field
from typing import Callable, Optional
import os

import numpy as np
from dataclasses_json import config, dataclass_json
from matplotlib.figure import Figure

from .data_formats import (
    CauseEffectData,
    ExperimentMetaData,
)
from .concentration_response_fits import (
    CRF_Settings,
    ConcentrationResponsePrediction,
    Transforms,
    concentration_response_fit,
)
from .helpers import Predicted_LCs, compute_lc, compute_lc_from_curve, ll5
from .io import make_np_config
from .plotting import plot_sam_prediction
from .stress_survival_conversion import stress_to_survival, survival_to_stress
from copy import deepcopy


@dataclass_json
@dataclass
class SAM_Settings:
    """Settings for configuring the SAM (Stress Addition Model) used in concentration-response predictions."""

    #: If `True`, normalizes survival values based on environmental stress.
    normalize_survival_for_stress_conversion: bool = False

    #: Defines the formula used to calculate environmental stress. Supported options:
    #: - `"div"`: `additional_stress = stressor / control`
    #: - `"subtract"`: `additional_stress = 1 - (control - stressor)`
    #: - `"only_stress"`: `additional_stress = 1 - stressor`
    #: - `"stress_sub"`: Subtracts transformed control stress from transformed stressor stress.
    stress_form: str = "div"

    #: Transformation function applied to regression data prior to model fitting.
    transform: "Transforms" = field(
        default=Transforms.williams_and_linear_interpolation,
        metadata=config(
            encoder=lambda t: t.__name__.replace("transform_", ""),
            decoder=lambda t: getattr(Transforms, t),
        ),
    )

    #: q Parameter of beta distribution for survival to stress and vice versa conversions
    beta_q: float = 3.2

    #: p Parameter of beta distribution for survival to stress and vice versa conversions
    beta_p: float = 3.2

    #: Controls which library is used for DoseResponse Curve fitting. Either scipy for scipy.optimize.curve_fit or lmcurce for using https://github.com/MockaWolke/py_lmcurve_ll5
    curve_fit_lib: str = "scipy"

    fix_f_parameter_ll5: Optional[float] = None

    keep_co_stressor_f_parameter_free: bool = False

    f_param_modifier_pre_sam: Callable = lambda x: x
    e_param_modifier_pre_sam: Callable = lambda x: x

    def __post_init__(
        self,
    ):
        self.stress_to_survival: Callable = lambda x: stress_to_survival(
            x, p=self.beta_p, q=self.beta_q
        )
        self.survival_to_stress: Callable = lambda x: survival_to_stress(
            x, p=self.beta_p, q=self.beta_q
        )


@dataclass_json
@dataclass
class SAMPrediction:
    #: Fitted concentration-response prediction for the control data.
    control: ConcentrationResponsePrediction

    #: Fitted concentration-response prediction for the co-stressor data.
    co_stressor: ConcentrationResponsePrediction

    #: Array of survival predictions from the SAM model.
    predicted_survival: np.ndarray = make_np_config()

    #: Array of general stress values corresponding to the predicted survival.
    predicted_general_stress: np.ndarray = make_np_config()

    #: The additional stress assumed by the SAM model.
    assumed_additional_stress: float

    #: The range of the data. (normally 100 -> 100%, sometimes also 1 -> 100% or deviating values)
    max_survival: float

    settings: SAM_Settings

    new_model: Callable

    def plot(self, with_lcs: bool = True, title: Optional[str] = None) -> Figure:
        """
        Plots the SAM prediction with optional lethal concentration (LC) indicators.

        Parameters:
            with_lcs (bool): Whether to include LC indicators in the plot.
            title (Optional[str]): Title for the plot.

        Returns:
            Figure: A matplotlib figure containing the plot.
        """
        lcs = (
            get_sam_lcs(
                stress_fit=self.co_stressor,
                sam_sur=self.predicted_survival,
                max_survival=self.max_survival,
            )
            if with_lcs
            else None
        )
        return plot_sam_prediction(
            self, lcs=lcs, survival_max=self.max_survival, title=title
        )

    def save_to_file(self, file_path: str) -> None:
        """
        Saves the SAM prediction to a JSON file.

        Parameters:
            file_path (str): The file path to save the JSON data.
        """
        with open(file_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, file_path: str) -> "SAMPrediction":
        """
        Loads a SAM prediction from a JSON file.

        Parameters:
            file_path (str): The file path to load the JSON data from.

        Returns:
            SAMPrediction: The loaded SAM prediction.
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"Can't find file at {file_path}")

        with open(file_path, "r") as f:
            return cls.from_json(f.read())


NEW_STANDARD = SAM_Settings(
    normalize_survival_for_stress_conversion=False,
    stress_form="stress_sub",
)
STANDARD_SAM_SETTING = SAM_Settings(
    normalize_survival_for_stress_conversion=True,
    stress_form="div",
    fix_f_parameter_ll5=1.0,
)


def generate_sam_prediction(
    control_data: CauseEffectData,
    co_stressor_data: CauseEffectData,
    meta: Optional[ExperimentMetaData] = None,
    max_survival: Optional[float] = None,
    settings: SAM_Settings = STANDARD_SAM_SETTING,
) -> SAMPrediction:
    """Computes survival and stress predictions based on a main control series and a stressor series
    using the Stress Addition Model (SAM).

    Parameters:
        control_data (CauseEffectData): Concentration-response data for the control group.
        co_stressor_data (CauseEffectData): Concentration-response data for the stressor condition.
        meta (Optional[ExperimentMetaData]): Metadata, used to infer max survival if not provided.
        max_survival (Optional[float]): Maximum survival rate. Overrides `meta.max_survival` if given.
        settings (SAM_Settings): Configuration settings for SAM. Controls stress computation formula,
            normalization, and additional adjustments.

    Returns:
        SAMPrediction
    Example:
        ```python
        prediction = generate_sam_prediction(control_data, co_stressor_data, settings=SAM_Settings(stress_form="div"))
        ```
    """

    if len(control_data.survival_rate) != len(co_stressor_data.survival_rate):
        raise ValueError("both data series must be of same length")
    if not np.all(control_data.concentration == co_stressor_data.concentration):
        raise ValueError("control and costressor must have identical concentrations")

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
        fix_f_parameter_ll5=settings.fix_f_parameter_ll5,
    )

    main_fit = concentration_response_fit(control_data, cfg=crf_cfg)

    # special fit because of keep_co_stressor_f_parameter_free
    stressor_crf_cgf = deepcopy(crf_cfg)
    if settings.keep_co_stressor_f_parameter_free:
        stressor_crf_cgf.fix_f_parameter_ll5 = None
    stressor_fit = concentration_response_fit(co_stressor_data, cfg=stressor_crf_cgf)

    additional_stress = compute_additional_stress(
        control_first_surivival_normed=main_fit.optim_param["d"],
        co_stressor_first_surivival_normed=stressor_fit.optim_param["d"],
        beta_p=settings.beta_p,
        beta_q=settings.beta_q,
        stress_form=settings.stress_form,
    )

    # specific calc with f modifier
    new_f = settings.f_param_modifier_pre_sam(main_fit.optim_param["f"])
    new_e = settings.e_param_modifier_pre_sam(main_fit.optim_param["e"])

    def new_model(x):
        return ll5(
            x,
            b=main_fit.optim_param["b"],
            c=main_fit.optim_param["c"],
            d=main_fit.optim_param["d"],
            e=new_e,
            f=new_f,
        )

    pred_survival = new_model(main_fit.concentration)
    if settings.normalize_survival_for_stress_conversion:
        stress_curve = survival_to_stress(
            pred_survival / main_fit.optim_param["d"],
            p=settings.beta_p,
            q=settings.beta_q,
        )
    else:
        stress_curve = survival_to_stress(
            pred_survival, p=settings.beta_p, q=settings.beta_q
        )

    predicted_stress_curve = np.clip(stress_curve + additional_stress, 0, 1)

    if settings.normalize_survival_for_stress_conversion:
        predicted_survival_curve = (
            settings.stress_to_survival(predicted_stress_curve)
            * main_fit.optim_param["d"]
            * max_survival
        )
    else:
        predicted_survival_curve = (
            settings.stress_to_survival(predicted_stress_curve) * max_survival
        )

    return SAMPrediction(
        main_fit,
        stressor_fit,
        predicted_survival_curve,
        predicted_stress_curve,
        additional_stress,
        max_survival,
        settings=settings,
        new_model=pred_survival,
    )


def compute_additional_stress(
    control_first_surivival_normed: CauseEffectData,
    co_stressor_first_surivival_normed: CauseEffectData,
    stress_form: str,
    beta_p: float,
    beta_q: float,
) -> float:
    sur2stress = lambda x: survival_to_stress(x, p=beta_p, q=beta_q)

    if stress_form == "div":
        return sur2stress(
            co_stressor_first_surivival_normed / control_first_surivival_normed
        )
    elif stress_form == "substract":
        return sur2stress(
            1 - (control_first_surivival_normed - co_stressor_first_surivival_normed)
        )
    elif stress_form == "only_stress":
        return sur2stress(1 - co_stressor_first_surivival_normed)
    elif stress_form == "stress_sub":
        return sur2stress(control_first_surivival_normed) - sur2stress(
            co_stressor_first_surivival_normed
        )
    else:
        raise ValueError(f"Unknown stress form '{stress_form}'")


def get_sam_lcs(
    stress_fit: ConcentrationResponsePrediction,
    sam_sur: np.ndarray,
    max_survival: float,
) -> Predicted_LCs:
    """
    Calculates lethal concentrations (LC10 and LC50) for stress and SAM predictions.

    Parameters:
        stress_fit (ModelPredictions): Fitted model predictions for the stressor.
        sam_sur (np.ndarray): Survival values from SAM predictions.

    Returns:
        Predicted_LCs: Lethal concentrations for both stress (LC10, LC50) and SAM predictions.
    """
    stress_lc10 = compute_lc(optim_param=stress_fit.optim_param, lc=10)
    stress_lc50 = compute_lc(optim_param=stress_fit.optim_param, lc=50)

    sam_lc10 = compute_lc_from_curve(
        stress_fit.concentration,
        sam_sur,
        lc=10,
        survival_max=max_survival,
        c0=stress_fit.optim_param["d"],
    )
    sam_lc50 = compute_lc_from_curve(
        stress_fit.concentration,
        sam_sur,
        lc=50,
        survival_max=max_survival,
        c0=stress_fit.optim_param["d"],
    )

    return Predicted_LCs(
        stress_lc10=stress_lc10,
        stress_lc50=stress_lc50,
        sam_lc10=sam_lc10,
        sam_lc50=sam_lc50,
    )
