import os
from dose_reponse_fit import (
    dose_response_fit,
    ModelPredictions,
    StandardSettings,
    survival_to_stress,
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

def compute_additional_stress(stressor_series: DoseResponseSeries, survival_max : float):

    return survival_to_stress(stressor_series.survival_rate[0] / survival_max)


def sam_prediction(
    main_series: DoseResponseSeries,
    stressor_series: DoseResponseSeries,
    meta: ExperimentMetaData,
):

    main_fit = dose_response_fit(
        main_series, cfg=StandardSettings(survival_max=meta.max_survival)
    )
    stressor_fit = dose_response_fit(
        stressor_series, cfg=StandardSettings(survival_max=meta.max_survival)
    )


    additional_stress = compute_additional_stress(stressor_series=stressor_series, survival_max = meta.max_survival)

    predicted_stress_curve = np.minimum(main_fit.stress_curve + additional_stress, 1)

    predicted_survival_curve = stress_to_survival(predicted_stress_curve) * meta.max_survival

    return main_fit, stressor_fit, predicted_survival_curve, predicted_stress_curve

@dataclass
class Predicted_LCs:
    stress_lc10 : float
    stress_lc50 : float
    sam_lc10 : float
    sam_lc50 : float
    
    
def get_sam_lcs(
        main_series: DoseResponseSeries,
    stressor_series: DoseResponseSeries,
    meta: ExperimentMetaData,
):

    main_fit, stress_fit, sam_sur, sam_stress = sam_prediction(main_series, stressor_series, meta)
    max_val = find_lc_99_max(stress_fit.model)

    stress_lc10 = compute_lc(stress_fit.model, 10, 1e-7, max_val)
    stress_lc50 = compute_lc(stress_fit.model, 50, 1e-7, max_val)

    sam_lc10 = compute_lc_from_curve(main_fit.concentration_curve, sam_sur, llc=10, survival_max=meta.max_survival)
    sam_lc50 = compute_lc_from_curve(main_fit.concentration_curve, sam_sur, llc=50, survival_max=meta.max_survival)
    
    return Predicted_LCs(stress_lc10=stress_lc10, stress_lc50=stress_lc50, sam_lc10=sam_lc10, sam_lc50=sam_lc50)