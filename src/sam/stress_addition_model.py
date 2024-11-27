from .dose_reponse_fit import (
    dose_response_fit,
    ModelPredictions,
    DRF_Settings,
    survival_to_stress,
    Transforms,
)
from .helpers import (
    compute_lc,
    Predicted_LCs,
    compute_lc_from_curve,
    detect_hormesis_index,
)
import numpy as np
from .data_formats import (
    ExperimentMetaData,
    DoseResponseSeries,
)
from .system_stress import calc_system_stress
from .stress_survival_conversion import stress_to_survival, survival_to_stress
from dataclasses import dataclass
from warnings import warn 
from typing import Optional, Callable

@dataclass
class SAM_Settings:
    """Settings for configuring the SAM (Stress Addition Model) used in dose-response predictions.
    """
    
    #: If `True`, normalizes survival values based on environmental stress.
    param_d_norm: bool = False
    
    #: Defines the formula used to calculate environmental stress. Supported options:
    #: - `"div"`: `additional_stress = stressor / control`
    #: - `"subtract"`: `additional_stress = 1 - (control - stressor)`
    #: - `"only_stress"`: `additional_stress = 1 - stressor`
    #: - `"stress_sub"`: Subtracts transformed control stress from transformed stressor stress.
    stress_form: str = "div"
    
    #: Adds a constant factor to `additional_stress` for modifying survival predictions (default is `1`).
    stress_intercept_in_survival: float = 1
    
    #: Upper bound on survival for control samples, useful for steep dose-response curves (default is `1`).
    max_control_survival: float = 1
    
    #: Transformation function applied to regression data prior to model fitting.
    transform: 'Transforms' = Transforms.williams_and_linear_interpolation
    
    #: Mapping function from stress values to survival values, defaulting to `stress_to_survival(x, 3.2, 3.2)`.
    stress_to_survival: Callable = lambda x: stress_to_survival(x, 3.2, 3.2)
    
    #: Mapping function from survival values to stress values, defaulting to `survival_to_stress(x, 3.2, 3.2)`.
    survival_to_stress: Callable = lambda x: survival_to_stress(x, 3.2, 3.2)

    
    #: If `True`, estimates system stress removes it from predictions.
    cancel_system_stress: bool = False
    
    #: Maximum Value of removed system stress if `cancel_system_stress = True`
    max_system_stress : float = 0.1

NEW_STANDARD = SAM_Settings(
    param_d_norm=False,
    stress_form="stress_sub",
    stress_intercept_in_survival=0.9995,
    max_control_survival=0.995,
)
STANDARD_SAM_SETTING = SAM_Settings(
    param_d_norm=True,
    stress_form="div",
    stress_intercept_in_survival=1,
    max_control_survival=1,
)

def sam_prediction(
    main_series: DoseResponseSeries,
    stressor_series: DoseResponseSeries,
    meta: Optional[ExperimentMetaData] = None,
    max_survival: Optional[float] = None,
    hormesis_index: Optional[int] = None,
    settings: SAM_Settings = STANDARD_SAM_SETTING,
):
    """Computes survival and stress predictions based on a main control series and a stressor series 
    using the Stress Addition Model (SAM).

    Parameters:
        main_series (DoseResponseSeries): Dose-response data for the control group.
        stressor_series (DoseResponseSeries): Dose-response data for the stressor condition.
        meta (Optional[ExperimentMetaData]): Metadata, used to infer max survival if not provided.
        max_survival (Optional[float]): Maximum survival rate. Overrides `meta.max_survival` if given.
        hormesis_index (Optional[int]): Index indicating hormetic effect. If `cancel_system_stress` is 
            enabled in settings, detects or uses this index to adjust predictions.
        settings (SAM_Settings): Configuration settings for SAM. Controls stress computation formula,
            normalization, and additional adjustments.

    Returns:
        Tuple: Contains:
            - `main_fit` (ModelPredictions): Predictions for the control data.
            - `stressor_fit` (ModelPredictions): Predictions for the stressor data.
            - `predicted_survival_curve` (np.ndarray): Predicted survival curve.
            - `predicted_stress_curve` (np.ndarray): Predicted stress curve.
            - `additional_stress` (float): Computed environmental stress level.

    Example:
        ```python
        prediction = sam_prediction(main_series, stressor_series, settings=SAM_Settings(stress_form="div"))
        ```
    """
    
    if max_survival is None and meta is None:
        raise ValueError(
            "Either `max_survival` or `meta` with a defined `max_survival` must be provided. "
            "Specify `meta` or directly set `max_survival` to proceed."
        )
    max_survival = meta.max_survival if max_survival is None else max_survival 


    dose_cfg = DRF_Settings(
        max_survival=max_survival,
        param_d_norm=settings.param_d_norm,
        stress_to_survival=settings.stress_to_survival,
        survival_to_stress=settings.survival_to_stress,
    )

    main_fit = dose_response_fit(main_series, cfg=dose_cfg)
    stressor_fit = dose_response_fit(stressor_series, cfg=dose_cfg)

    sur2stress = settings.survival_to_stress
    stress2sur = settings.stress_to_survival

    if settings.stress_form == "div":
        additional_stress = stressor_fit.optim_param["d"] / main_fit.optim_param["d"]
    elif settings.stress_form == "substract":
        additional_stress = 1 - (
            main_fit.optim_param["d"] - stressor_fit.optim_param["d"]
        )
    elif settings.stress_form == "only_stress":
        additional_stress = 1 - stressor_fit.optim_param["d"]
    elif settings.stress_form == "stress_sub":

        a = sur2stress(stressor_fit.optim_param["d"])

        control_survival = min(main_fit.optim_param["d"], settings.max_control_survival)

        b = sur2stress(control_survival)

        additional_stress = stress2sur(a - b)

    else:
        raise ValueError(f"Unknown stress form '{settings.stress_form}'")

    additional_stress = sur2stress(additional_stress) + sur2stress(
        settings.stress_intercept_in_survival
    )

    if not settings.cancel_system_stress:

        predicted_stress_curve = np.clip(
            main_fit.stress_curve + additional_stress, 0, 1
        )

    else:
        if hormesis_index is None:
            hormesis_index = detect_hormesis_index(main_series.survival_rate)
            warn(f"could not find hormesis index, detecting it to be {hormesis_index}")
            if hormesis_index is None:
                raise ValueError("Cant detect hormesis!")

        without_horm, system_stress = calc_system_stress(
            only_tox_series=main_series,
            dose_response_fit=main_fit,
            hormesis_index=hormesis_index,
            cfg=dose_cfg,
        )

        main_fit.cleaned_survival = without_horm(main_fit.concentrations) * max_survival
        main_fit.cleaned_stress = sur2stress(main_fit.cleaned_survival / max_survival)
        
        
        main_fit.pred_system_stress = system_stress(main_fit.concentrations)
        
        sys_stress_to_add = min(main_fit.pred_system_stress.max(), settings.max_system_stress)
        print(main_fit.pred_system_stress.max(), settings.max_system_stress, sys_stress_to_add)
        
        main_fit.modified_control_stress = np.clip(main_fit.cleaned_stress + sys_stress_to_add, 0, 1)
        main_fit.modified_control_surv = stress2sur(main_fit.modified_control_stress) * max_survival

        predicted_stress_curve = np.clip(
            main_fit.modified_control_stress + additional_stress, 0, 1
        )       

    if settings.param_d_norm:
        predicted_survival_curve = (
            stress2sur(predicted_stress_curve)
            * main_fit.optim_param["d"]
            * max_survival
        )
    else:
        predicted_survival_curve = (
            stress2sur(predicted_stress_curve) * max_survival
        )

    return (
        main_fit,
        stressor_fit,
        predicted_survival_curve,
        predicted_stress_curve,
        additional_stress,
    )

def get_sam_lcs(
    stress_fit: ModelPredictions,
    sam_sur: np.ndarray,
    meta: ExperimentMetaData,
)-> Predicted_LCs:
    """
    Calculates lethal concentrations (LC10 and LC50) for stress and SAM predictions.

    Parameters:
        stress_fit (ModelPredictions): Fitted model predictions for the stressor.
        sam_sur (np.ndarray): Survival values from SAM predictions.
        meta (ExperimentMetaData): Experiment metadata, providing max survival.

    Returns:
        Predicted_LCs: Lethal concentrations for both stress (LC10, LC50) and SAM predictions.
    """
    stress_lc10 = compute_lc(optim_param=stress_fit.optim_param, lc=10)
    stress_lc50 = compute_lc(optim_param=stress_fit.optim_param, lc=50)

    sam_lc10 = compute_lc_from_curve(
        stress_fit.concentrations,
        sam_sur,
        lc=10,
        survival_max=meta.max_survival,
        c0=stress_fit.optim_param["d"],
    )
    sam_lc50 = compute_lc_from_curve(
        stress_fit.concentrations,
        sam_sur,
        lc=50,
        survival_max=meta.max_survival,
        c0=stress_fit.optim_param["d"],
    )

    return Predicted_LCs(
        stress_lc10=stress_lc10,
        stress_lc50=stress_lc50,
        sam_lc10=sam_lc10,
        sam_lc50=sam_lc50,
    )
