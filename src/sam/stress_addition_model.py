from .dose_reponse_fit import (
    dose_response_fit,
    ModelPredictions,
    FitSettings,
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

@dataclass
class SAM_Setting:
    param_d_norm: bool = False
    transform: Transforms = Transforms.williams_and_linear_interpolation
    stress_form: str = "div"  # "only_stress" or "div" "substract"
    stress_intercept_in_survival: float = 1
    max_control_survival: float = 1
    stress_to_survival: int = lambda x: stress_to_survival(x, 3.2, 3.2)
    survival_to_stress: int = lambda x: survival_to_stress(x, 3.2, 3.2)
    cancel_system_stress: bool = False


NEW_STANDARD = SAM_Setting(
    param_d_norm=False,
    stress_form="stress_sub",
    stress_intercept_in_survival=0.9995,
    max_control_survival=0.995,
)
OLD_STANDARD = SAM_Setting(
    param_d_norm=True,
    stress_form="div",
    stress_intercept_in_survival=1,
    max_control_survival=1,
)


def sam_prediction(
    main_series: DoseResponseSeries,
    stressor_series: DoseResponseSeries,
    meta: ExperimentMetaData,
    settings: SAM_Setting = SAM_Setting(),
):

    dose_cfg = FitSettings(
        survival_max=meta.max_survival,
        param_d_norm=settings.param_d_norm,
        stress_to_survival=settings.stress_to_survival,
        survival_to_stress=settings.survival_to_stress,
    )

    main_fit = dose_response_fit(main_series, cfg=dose_cfg)
    stressor_fit = dose_response_fit(stressor_series, cfg=dose_cfg)

    sur2stress = lambda x: settings.survival_to_stress(x)
    stress2sur = lambda x: settings.stress_to_survival(x)

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

        if hasattr(meta, "hormesis_index") and meta.hormesis_index is not None:
            hormesis_index = meta.hormesis_index
        else:
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

        pred_system_stress = system_stress(main_fit.concentration_curve)
        main_fit.pred_system_stress = pred_system_stress
        pred_system_stress =  (pred_system_stress - pred_system_stress.max()) * -1

        predicted_stress_curve = np.clip(
            main_fit.stress_curve + additional_stress + pred_system_stress, 0, 1
        )

    if settings.param_d_norm:
        predicted_survival_curve = (
            stress2sur(predicted_stress_curve)
            * main_fit.optim_param["d"]
            * meta.max_survival
        )
    else:
        predicted_survival_curve = (
            stress2sur(predicted_stress_curve) * meta.max_survival
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
):

    stress_lc10 = compute_lc(optim_param=stress_fit.optim_param, lc=10)
    stress_lc50 = compute_lc(optim_param=stress_fit.optim_param, lc=50)

    sam_lc10 = compute_lc_from_curve(
        stress_fit.concentration_curve,
        sam_sur,
        lc=10,
        survival_max=meta.max_survival,
        c0=stress_fit.optim_param["d"],
    )
    sam_lc50 = compute_lc_from_curve(
        stress_fit.concentration_curve,
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
