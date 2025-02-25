from dataclasses import dataclass, field
from typing import Callable, Optional
import os

import numpy as np
from dataclasses_json import config, dataclass_json
from matplotlib.figure import Figure

from .concentration_addition import concentration_addition_prediction

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
from .helpers import Predicted_LCs, compute_lc, ll5, ll5_inv, LC_Verification_Error
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

    e_param_fac: float = 1.0

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

    effect_addition_prediction: np.ndarray = make_np_config()

    concentratation_addition_prediction: np.ndarray = make_np_config()

    normalize_survival_for_stress_conversion_factor: float

    def predict(self, concentration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        def new_model(x):
            return ll5(
                x,
                b=self.control.optim_param["b"],
                c=self.control.optim_param["c"],
                d=self.control.optim_param["d"],
                e=self.control.optim_param["e"] * self.settings.e_param_fac,
                f=self.control.optim_param["f"],
            )

        pred_survival = new_model(concentration)
        stress_curve = self.settings.survival_to_stress(
            pred_survival / self.normalize_survival_for_stress_conversion_factor,
        )

        predicted_stress_curve = np.clip(
            stress_curve + self.assumed_additional_stress, 0, 1
        )

        predicted_survival_curve = (
            self.settings.stress_to_survival(predicted_stress_curve)
            * self.normalize_survival_for_stress_conversion_factor
            * self.max_survival
        )
        return predicted_survival_curve, predicted_stress_curve

    def plot(
        self,
        with_lcs: bool = True,
        title: Optional[str] = None,
        inlcude_control_addition: bool = False,
    ) -> Figure:
        """
        Plots the SAM prediction with optional lethal concentration (LC) indicators.

        Parameters:
            with_lcs (bool): Whether to include LC indicators in the plot.
            title (Optional[str]): Title for the plot.

        Returns:
            Figure: A matplotlib figure containing the plot.
        """
        lcs = None
        if with_lcs:
            try:
                lcs = get_sam_lcs(self)
            except LC_Verification_Error as e:
                print(f"There was an error when revalidating the compiuted lcs:\n\t{e}")

        return plot_sam_prediction(
            self,
            lcs=lcs,
            survival_max=self.max_survival,
            title=title,
            inlcude_control_addition=inlcude_control_addition,
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


STANDARD_SAM_SETTING = SAM_Settings(
    normalize_survival_for_stress_conversion=True,
    stress_form="div",
    fix_f_parameter_ll5=1.0,
    keep_co_stressor_f_parameter_free=False,
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
    def new_model(x):
        return ll5(
            x,
            b=main_fit.optim_param["b"],
            c=main_fit.optim_param["c"],
            d=main_fit.optim_param["d"],
            e=main_fit.optim_param["e"] * settings.e_param_fac,
            f=main_fit.optim_param["f"],
        )

    normalization_factor = (
        main_fit.optim_param["d"]
        if settings.normalize_survival_for_stress_conversion
        else 1.0
    )

    def sam_pred(concentration: np.ndarray):
        pred_survival = new_model(concentration)
        stress_curve = settings.survival_to_stress(
            pred_survival / normalization_factor,
        )

        predicted_stress_curve = np.clip(stress_curve + additional_stress, 0, 1)

        predicted_survival_curve = (
            settings.stress_to_survival(predicted_stress_curve)
            * normalization_factor
            * max_survival
        )
        return predicted_survival_curve, predicted_stress_curve

    predicted_survival_curve, predicted_stress_curve = sam_pred(
        concentration=main_fit.concentration
    )

    # effect addition
    effect_addition_pred = (
        stressor_fit.optim_param["d"] / main_fit.optim_param["d"] * main_fit.survival
    )

    return SAMPrediction(
        main_fit,
        stressor_fit,
        predicted_survival_curve,
        predicted_stress_curve,
        additional_stress,
        max_survival,
        settings=settings,
        new_model=predicted_survival_curve,
        effect_addition_prediction=effect_addition_pred,
        normalize_survival_for_stress_conversion_factor=normalization_factor,
        concentratation_addition_prediction=concentration_addition_prediction(
            control_params=main_fit.optim_param,
            co_stressor_params=stressor_fit.optim_param,
            concentration=main_fit.concentration,
            max_survival=max_survival,
        ),
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


def get_sam_lcs(sam_prediction: SAMPrediction) -> Predicted_LCs:
    """
    Calculates lethal concentrations (LC10 and LC50) for stress and SAM predictions.

    Parameters:
        stress_fit (ModelPredictions): Fitted model predictions for the stressor.
        sam_sur (np.ndarray): Survival values from SAM predictions.

    Returns:
        Predicted_LCs: Lethal concentrations for both stress (LC10, LC50) and SAM predictions.
    """
    stress_fit = sam_prediction.co_stressor

    sc0 = stress_fit.optim_param["d"]

    stress_lc10 = compute_lc(optim_param=stress_fit.optim_param, lc=10)
    if not np.isclose(sc0 * 0.9, stress_fit.model(stress_lc10)):
        raise LC_Verification_Error(
            f"Stress LC10 calculation error: expected {sc0 * 0.9}, got {stress_fit.model(stress_lc10)}"
        )

    stress_lc50 = compute_lc(optim_param=stress_fit.optim_param, lc=50)
    if not np.isclose(sc0 * 0.5, stress_fit.model(stress_lc50)):
        raise LC_Verification_Error(
            f"Stress LC50 calculation error: expected {sc0 * 0.5}, got {stress_fit.model(stress_lc50)}"
        )

    sam_lc10 = compute_sam_lc(pred=sam_prediction, lc=10)
    sam_recomputed_10 = (
        sam_prediction.predict(sam_lc10)[0] / sam_prediction.max_survival
    )

    if not np.isclose(sc0 * 0.9, sam_recomputed_10):
        raise LC_Verification_Error(
            f"SAM LC10 calculation error: expected {sc0 * 0.9}, got {sam_recomputed_10}"
        )

    sam_lc50 = compute_sam_lc(pred=sam_prediction, lc=50)
    sam_recomputed_50 = (
        sam_prediction.predict(sam_lc50)[0] / sam_prediction.max_survival
    )

    if not np.isclose(sc0 * 0.5, sam_recomputed_50):
        raise LC_Verification_Error(
            f"SAM LC50 calculation error: expected {sc0 * 0.5}, got {sam_recomputed_50}"
        )

    return Predicted_LCs(
        stress_lc10=stress_lc10,
        stress_lc50=stress_lc50,
        sam_lc10=sam_lc10,
        sam_lc50=sam_lc50,
    )


def compute_sam_lc(pred: SAMPrediction, lc: float) -> float:
    pa = pred.control.optim_param

    # value in refrence to co stressor c0
    val = (1 - (lc / 100)) * pred.co_stressor.optim_param["d"]

    fac = pa["d"] if pred.settings.normalize_survival_for_stress_conversion else 1.0

    stress = pred.settings.survival_to_stress(val / fac)

    surv = (
        pred.settings.stress_to_survival(
            stress - pred.assumed_additional_stress,
        )
        * fac
    )

    from copy import deepcopy

    pa_copy = deepcopy(pa)
    pa_copy["e"] *= pred.settings.e_param_fac

    return ll5_inv(surv=surv, **pa_copy)
