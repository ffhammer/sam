import os
from sam import chdir_to_repopath

chdir_to_repopath()
from sam.dose_reponse_fit import (
    dose_response_fit,
    ModelPredictions,
    survival_to_stress,
    DRF_Settings,
)
import pandas as pd
import numpy as np
from sam.data_formats import load_files
from sam.system_stress import pred_surv_without_hormesis
from sam.helpers import (
    detect_hormesis_index,
    pad_c0,
    weibull_2param_inverse,
)
from sam.data_formats import ExperimentData


def predict_cleaned_curv(data: ExperimentData):
    """ "Predicts the cleaned Curv"""
    concentration = pad_c0(data.main_series.concentration).copy()
    survival_tox_observerd = np.copy(
        data.main_series.survival_rate / data.meta.max_survival
    )

    if data.meta.hormesis_concentration is None:

        hormesis_index = detect_hormesis_index(survival_tox_observerd)

        if hormesis_index is None:
            hormesis_index = 1

    else:
        hormesis_index = np.argwhere(
            data.meta.hormesis_concentration == data.main_series.concentration
        )[0, 0]

    func, _, popt = pred_surv_without_hormesis(
        concentration=concentration,
        surv_withhormesis=survival_tox_observerd,
        hormesis_index=hormesis_index,
    )

    return func, hormesis_index, popt


def create_dose_response_fits_frame() -> pd.DataFrame:
    dfs = []

    for path, data in load_files():

        meta = data.meta
        res: ModelPredictions = dose_response_fit(
            data.main_series,
            DRF_Settings(param_d_norm=True, max_survival=meta.max_survival),
        )

        cleaned_func, _, popt = predict_cleaned_curv(data)

        inverse = lambda x: weibull_2param_inverse(x, *popt)

        def find_lc(lc):
            lc = 1 - lc / 100
            return inverse(lc)

        lc1 = find_lc(1)
        lc99 = find_lc(99)
        dfs.append(
            {
                "title": os.path.split(path[:-5])[1],
                "chemical": meta.chemical,
                "Organism": meta.organism,
                "model": res,
                "cleaned_func": cleaned_func,
                "lc1": lc1,
                "lc99": lc99,
                "Name": meta.title,
                "Duration": int(meta.days),
                "Experiment": meta.path.parent.name,
            }
        )

    df = pd.DataFrame(dfs)

    def compute_normalised_curve(model: ModelPredictions):

        if np.isnan(model.lc1):
            print("nan")
            model.lc1 = 0.0
        x = np.linspace(model.lc1, model.lc99, 1000)

        return model.model(x) * 100

    def compute_cleaned_curve(row):

        x = np.linspace(row.lc1, row.lc99, 1000)

        return row.cleaned_func(x) * 100

    df["normed_curves"] = df.model.apply(compute_normalised_curve)
    df["stress"] = df.normed_curves.apply(lambda x: survival_to_stress(x / 100))

    df["cleaned_curves"] = df.apply(compute_cleaned_curve, axis=1)
    df["cleaned_stress"] = df.cleaned_curves.apply(
        lambda x: survival_to_stress(x / 100)
    )

    return df
