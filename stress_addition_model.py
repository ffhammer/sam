import os
from dose_reponse_fit import (
    dose_response_fit,
    ModelPredictions,
    StandardSettings,
    survival_to_stress,
    Transforms
)
from helpers import compute_lc, find_lc_99_max, compute_lc_from_curve
from plotting import plot_fit_prediction
import pandas as pd
import numpy as np
import glob
from data_formats import (
    ExperimentData,
    ExperimentMetaData,
    DoseResponseSeries,
    read_data,
)
from stress_survival_conversion import stress_to_survival, survival_to_stress
import matplotlib.pyplot as plt
from dataclasses import dataclass
from helpers import Predicted_LCs




@dataclass
class SAM_Setting:
    beta_q : float = 3.2
    beta_p : float = 3.2
    param_d_norm : bool = False
    transform : Transforms = Transforms.williams_and_linear_interpolation
    stress_form : str = "only_stress" # "only_stress" or "div" "substract"
    stress_intercept_in_survival : float = 1
    max_control_survival : float = 1
    exponent : float = 1 
    sub : float = None


NEW_STANDARD = SAM_Setting(beta_p=3.2, beta_q=3.2, param_d_norm=False, stress_form= "stress_sub", stress_intercept_in_survival=0.9995, max_control_survival=0.995)
OLD_STANDARD = SAM_Setting(beta_p=3.2, beta_q=3.2, param_d_norm=True, stress_form= "div", stress_intercept_in_survival=1, max_control_survival=1)


def sam_prediction(
    main_series: DoseResponseSeries,
    stressor_series: DoseResponseSeries,
    meta: ExperimentMetaData,
    settings: SAM_Setting = SAM_Setting(),
):
    
    dose_cfg = StandardSettings(survival_max=meta.max_survival, beta_q=settings.beta_q, beta_p=settings.beta_p, param_d_norm=settings.param_d_norm)

    main_fit = dose_response_fit(
        main_series, cfg=dose_cfg
    )
    stressor_fit = dose_response_fit(
        stressor_series, cfg=dose_cfg
    )
    
    sur2stress = lambda x : survival_to_stress(x, p=settings.beta_p, q=settings.beta_q)
    stress2sur = lambda x : stress_to_survival(x, p=settings.beta_p, q=settings.beta_q)
    

    if settings.stress_form == "div":
        additional_stress = stressor_fit.optim_param["d"] / main_fit.optim_param["d"]
    elif settings.stress_form == "substract":
        additional_stress = 1 - (main_fit.optim_param["d"]  - stressor_fit.optim_param["d"])
    elif settings.stress_form == "only_stress":
        additional_stress = 1 - stressor_fit.optim_param["d"] 
    elif settings.stress_form == "stress_sub":
        
        a = sur2stress(stressor_fit.optim_param["d"])
        
        control_survival = min(main_fit.optim_param["d"], settings.max_control_survival)
        
        b = sur2stress(control_survival)
        
        additional_stress = stress2sur(a -b)
        
    else:
        raise ValueError(f"Unknown stress form '{settings.stress_form}'")
        
    additional_stress = sur2stress(additional_stress) + sur2stress(settings.stress_intercept_in_survival)
    
    
    if settings.sub is None:
        additional_stress = additional_stress ** settings.exponent
    else:
        exp = (settings.sub - additional_stress ) * settings.exponent
        additional_stress = additional_stress ** exp
        
        
    predicted_stress_curve = np.minimum(main_fit.stress_curve + additional_stress, 1)
    
        
    if settings.param_d_norm:
        predicted_survival_curve = stress2sur(predicted_stress_curve) * main_fit.optim_param["d"] * meta.max_survival
    else:
        predicted_survival_curve = stress2sur(predicted_stress_curve) * meta.max_survival


    return main_fit, stressor_fit, predicted_survival_curve, predicted_stress_curve, additional_stress


    
    
    
def get_sam_lcs(
        stress_fit : ModelPredictions,
        sam_sur : np.ndarray,
    meta: ExperimentMetaData,
):

    max_val = find_lc_99_max(stress_fit.model)

    stress_lc10 = compute_lc(stress_fit.model, 10, 1e-7, max_val)
    stress_lc50 = compute_lc(stress_fit.model, 50, 1e-7, max_val)

    sam_lc10 = compute_lc_from_curve(stress_fit.concentration_curve, sam_sur, llc=10, survival_max=meta.max_survival)
    sam_lc50 = compute_lc_from_curve(stress_fit.concentration_curve, sam_sur, llc=50, survival_max=meta.max_survival)
    
    return Predicted_LCs(stress_lc10=stress_lc10, stress_lc50=stress_lc50, sam_lc10=sam_lc10, sam_lc50=sam_lc50)